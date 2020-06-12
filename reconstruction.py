#! /home/zliu0/usr/anaconda3/envs/tfgpu/bin/python
import tensorflow as tf 
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model


from functools import partial

import numpy as np 
import os
from util import save2img
import sys, os, time, argparse, shutil, scipy, h5py, glob
from models_inpainting import tomogan_disc as make_discriminator_model, recon_generator_model, fc_model  # import a disc model

from data_processor import bkgdGen, gen_train_batch_bg, get1batch4test, get_train_slice, prepare_training_array
import datetime
from sn import SpectralNormalization
from pathlib import Path

from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.layers.projection_2d import parallel_projection2d
from pyronn.ct_reconstruction.layers.backprojection_2d import parallel_backprojection2d
from pyronn.ct_reconstruction.helpers.misc import generate_sinogram as sino_helper
from pyronn.ct_reconstruction.helpers.misc import generate_reco as reco_helper
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.filters import filters

from pyronn.ct_reconstruction.geometry.geometry_parallel_3d import GeometryParallel3D
from pyronn.ct_reconstruction.helpers.filters import filters


parser = argparse.ArgumentParser(description='encode sinogram image.')
parser.add_argument('-gpus',  type=str, default="0", help='list of visiable GPUs')
parser.add_argument('-expName', type=str, default="Reconstruction", required=False, help='Experiment name')
parser.add_argument('-lpixel', type=float, default=1., help='lambda pixel loss')
parser.add_argument('-ladv', type=float, default=0, help='lambda adv') # discriminator loss is not used for final reconstruction, only for training
parser.add_argument('-lunet', type=int, default=9, help='Unet layers')
parser.add_argument('-depth', type=int, default=3, help='input number of channels')
parser.add_argument('-itg', type=int, default=1, help='iterations for G')
parser.add_argument('-itd', type=int, default=1, help='iterations for D')
parser.add_argument('-n_proj', type=int, default=256, help='number of projections taken for sparse sinogram')
parser.add_argument('-n_proj_inpaint', type=int, default=1024, help='number of projections for inpainted sinogram')
parser.add_argument('-rthresh', type=float, default=0.02, help='MAE threshold for reconstruction')
parser.add_argument('-recon', type=str, default='./test_dataset/series_5-net-Pt-filt_hdr0_SIRT30i_1n.h5', required=True, help='path to reconstruction file')


args, unparsed = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.99

@tf.function 
def normalize_tensor(x): 
    return ((x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x)) -0.5) / 0.5 

mb_size = 4
img_size = 1024
# in_depth = args.depth
in_depth = 1

# Volume Parameters:
volume_size = img_size
volume_shape = [volume_size, volume_size]
volume_spacing = [1, 1]

# Detector Parameters:
detector_shape = img_size
detector_spacing = 1

# Trajectory Parameters:
number_of_projections = args.n_proj #256
angular_range = np.pi 

number_of_projections_inpainted = args.n_proj_inp #1024
# angular_range_inpainted = np.radians(270)
angular_range_inpainted = np.pi 

disc_iters, gene_iters = args.itd, args.itg
lambda_pixel, lambda_adv = args.lpixel, args.ladv

recon_iter = 400 # maximum number of iterations per slice recontruction
it_per_slice = 201
save_every = 100
threshold_pixel_loss = args.rthresh # range <0.01; 0.05> works

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

itr_out_dir = args.expName + '/' + current_time + '/' + 'itr_out'
final_recon_out_dir = args.expName + '/' + current_time + '/' + 'Final_recon'

Path(itr_out_dir).mkdir(parents=True, exist_ok=False)
Path(final_recon_out_dir).mkdir(parents=True, exist_ok=False)


#Mask inpainting
np_mask = np.zeros((number_of_projections_inpainted, img_size))
np_mask[0::(number_of_projections_inpainted//number_of_projections)] = 1

# #Mask missing wedge
# np_mask = np.ones((number_of_projections_inpainted, img_size))
# np_mask[206:306] = 0

tf_mask = tf.convert_to_tensor(np_mask, np.float32)

tf_mask = tf.expand_dims(tf_mask, axis = 0)
tf_mask = tf.expand_dims(tf_mask, axis = -1)

generator = recon_generator_model(input_shape=(number_of_projections_inpainted, img_size, 2*in_depth+1), dilation_rate=1, n_layers_unet=args.lunet)

generator.load_weights('./weights/8_rows_inpainting.h5', by_name = True, skip_mismatch=False)

discriminator = make_discriminator_model(input_shape=(number_of_projections, img_size, 2*in_depth+1))

fc_inverse_radon = fc_model(input_shape=(number_of_projections_inpainted, img_size, 1), img_size=img_size)

#Workable
def gradient_penalty(real, fake, disc):
    alpha = tf.random.uniform([mb_size, 1, 1, 1], 0., 1.)
    diff = fake - real
    inter = real + (alpha * diff)
    with tf.GradientTape() as t:
        t.watch(inter)
        pred = disc(inter)
    grad = t.gradient(pred, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])) #the only difference is axis
    gp = tf.reduce_mean((slopes - 1.)**2)
    return gp

def d_loss_fn(f_logit, r_logit):
    f_loss = tf.reduce_mean(f_logit)
    r_loss = tf.reduce_mean(r_logit)
    return f_loss - r_loss

def g_loss_fn(f_logit):
    f_loss = -tf.reduce_mean(f_logit)
    return f_loss

gen_optimizer  = tf.optimizers.Adam(1e-4) 
disc_optimizer = tf.optimizers.Adam(4e-4) 

train_gen_log_dir = './logs/gradient_tape/' + current_time + '/generator_train'

train_dis_log_dir = './logs/gradient_tape/' + current_time + '/discriminator_train'

if os.path.isdir('./logs/gradient_tape/'): 
    shutil.rmtree('./logs/gradient_tape/')

train_gen_summary_writer = tf.summary.create_file_writer(train_gen_log_dir)

train_dis_summary_writer = tf.summary.create_file_writer(train_dis_log_dir)

##Workable 2D, input 1 slice
geometry = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range)
geometry.set_trajectory(circular_trajectory.circular_trajectory_2d(geometry))

geometry_inpainted = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections_inpainted, angular_range_inpainted)
geometry_inpainted.set_trajectory(circular_trajectory.circular_trajectory_2d(geometry_inpainted))

reco_filter = filters.ram_lak_2D(geometry)
epoch = 0 

for dataset_path in [args.recon, ]: 
    #['./test_dataset/series_5-area_0_SIRT30i_2n.h5', ]:# ['./test_dataset/20170710.h5', ] #['./test_dataset/clean4test.h5', ] ['./test_dataset/phantom_00016_recon.h5', ] ['./dataset/TIQ.h5', ]
    #Get and prepare training array
    X = prepare_training_array(dataset_path, img_size)
    for idx in range(0, X.shape[0], 1): #for biological TEM tomo
    # for idx in range(X.shape[0]):
        tf.print('Proccessed_index:', idx, output_stream=sys.stdout)
        if idx == 0:
            original_sinogram_0 = parallel_projection2d(np.expand_dims(X[0], axis=0), geometry)
            inpainted_sinogram_0 = parallel_projection2d(np.expand_dims(X[0], 0), geometry_inpainted)
            
        else:
            original_sinogram_0 = parallel_projection2d(np.expand_dims(X[idx-1], axis=0), geometry)
            inpainted_sinogram_0 = parallel_projection2d(np.expand_dims(X[idx-1], 0), geometry_inpainted)
        
        original_sinogram_0 = tf.keras.layers.LayerNormalization()(original_sinogram_0)
        inpainted_sinogram_0 = tf.keras.layers.LayerNormalization()(inpainted_sinogram_0)

        original_sinogram_0 = tf.expand_dims(original_sinogram_0, axis = -1)
        inpainted_sinogram_0 = tf.expand_dims(inpainted_sinogram_0, axis = -1)

        original_sinogram_1 = parallel_projection2d(np.expand_dims(X[idx], axis=0), geometry)
        original_sinogram_1 = tf.keras.layers.LayerNormalization()(original_sinogram_1)
        original_sinogram_1 = tf.expand_dims(original_sinogram_1, axis = -1)

        inpainted_sinogram_1 = parallel_projection2d(np.expand_dims(X[idx], axis=0), geometry_inpainted)
        inpainted_sinogram_1 = tf.keras.layers.LayerNormalization()(inpainted_sinogram_1)
        inpainted_sinogram_1 = tf.expand_dims(inpainted_sinogram_1, axis = -1)

        if idx == (X.shape[0]-1):
            original_sinogram_2 = parallel_projection2d(np.expand_dims(X[idx], axis=0), geometry)
            inpainted_sinogram_2 = parallel_projection2d(np.expand_dims(X[idx], axis=0), geometry_inpainted)
        else:
            original_sinogram_2 = parallel_projection2d(np.expand_dims(X[idx+1], axis=0), geometry)
            inpainted_sinogram_2 = parallel_projection2d(np.expand_dims(X[idx+1], axis=0), geometry_inpainted)

        original_sinogram_2 = tf.keras.layers.LayerNormalization()(original_sinogram_2)
        inpainted_sinogram_2 = tf.keras.layers.LayerNormalization()(inpainted_sinogram_2)

        original_sinogram_2 = tf.expand_dims(original_sinogram_2, axis = -1)
        inpainted_sinogram_2 = tf.expand_dims(inpainted_sinogram_2, axis = -1)

        original_sinogram = tf.concat([original_sinogram_0, original_sinogram_1, original_sinogram_2], axis = -1)
        
        inpainted_sinogram = tf.concat([inpainted_sinogram_0, inpainted_sinogram_1, inpainted_sinogram_2], axis = -1)
        inpainted_sinogram_masked = tf_mask * inpainted_sinogram
        
        inpainted_sinogram_0_masked = tf_mask * inpainted_sinogram_0
        inpainted_sinogram_1_masked = tf_mask * inpainted_sinogram_1
        inpainted_sinogram_2_masked = tf_mask * inpainted_sinogram_2 
    

        gen_prediction, _ = generator.predict(inpainted_sinogram_masked)

        loss_pixel = tf.constant([10.**10])
        while tf.reduce_mean(loss_pixel) > tf.constant([threshold_pixel_loss], dtype=tf.float32): 
            time_git_st = time.time()
            for _ge in range(gene_iters):
                with tf.GradientTape() as gen_tape:
                    gen_tape.watch(fc_inverse_radon.trainable_variables) #watch gen tape for fully connected model only
                    recon_imgs = fc_inverse_radon(gen_prediction, training=True)

                    recon_imgs = tf.squeeze(recon_imgs, axis=-1, name=None)

                    generated_sinogram = parallel_projection2d(recon_imgs, geometry) 
                    generated_sinogram = tf.keras.layers.LayerNormalization()(generated_sinogram)
                    generated_sinogram = tf.expand_dims(generated_sinogram, axis = -1)

                    generated_sinogram_inpainted = parallel_projection2d(recon_imgs, geometry_inpainted) 
                    generated_sinogram_inpainted = tf.keras.layers.LayerNormalization()(generated_sinogram_inpainted)
                    generated_sinogram_inpainted = tf.expand_dims(generated_sinogram_inpainted, axis = -1)

                    generated_sinogram_3D = tf.concat([original_sinogram_0, generated_sinogram, original_sinogram_2], axis = -1)

                    disc_fake_o = discriminator(generated_sinogram_3D, training=True)

                    loss_pixel = tf.keras.losses.MAE(gen_prediction, generated_sinogram_inpainted)

                    loss_adv = g_loss_fn(disc_fake_o)

                    gen_loss = lambda_adv * loss_adv + lambda_pixel * loss_pixel
                    gen_loss = tf.expand_dims(gen_loss, axis = -1)

                gen_gradients = gen_tape.gradient(gen_loss, fc_inverse_radon.trainable_variables) #update weights for FC only
                gen_optimizer.apply_gradients(zip(gen_gradients, fc_inverse_radon.trainable_variables))

                # gen_gradients = gen_tape.gradient(gen_loss, gen_model_comb.trainable_variables) #Update weights for generator and FC: worse performance
                # gen_optimizer.apply_gradients(zip(gen_gradients, gen_model_comb.trainable_variables))

            if epoch < 10:
                tf.print('gen_elapse: {}s/itr'.format((time.time() - time_git_st)/gene_iters, ), output_stream=sys.stdout)

            with train_gen_summary_writer.as_default():
                tf.summary.scalar('generator_loss', tf.keras.backend.mean(gen_loss), step=epoch)
                tf.summary.scalar('generator_loss_pixel', tf.keras.backend.mean(lambda_pixel * loss_pixel), step=epoch)
                tf.summary.scalar('generator_loss_adv', tf.keras.backend.mean(loss_adv*lambda_adv), step=epoch)

            time_dit_st = time.time()

            for _dis in range(disc_iters):
                with tf.GradientTape() as disc_tape:
                    disc_tape.watch(discriminator.trainable_variables)
                   
                    disc_fake_o = discriminator(generated_sinogram_3D, training=True)
                    disc_real_o = discriminator(original_sinogram, training=True)

                    gp = gradient_penalty(original_sinogram, generated_sinogram_3D, partial(discriminator, training=True))

                    disc_loss = d_loss_fn(disc_fake_o, disc_real_o) + 10 * gp

                disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

            if epoch < 10:
                tf.print('disc_elapse: {}s/itr'.format((time.time() - time_dit_st)/disc_iters, ), output_stream=sys.stdout)

            with train_dis_summary_writer.as_default():
                tf.summary.scalar('discriminator_loss', tf.reduce_mean(disc_loss), step=epoch)
                tf.summary.scalar('discriminator_loss_real', tf.reduce_mean(disc_real_o), step=epoch)
                tf.summary.scalar('discriminator_loss_fake', tf.reduce_mean(disc_fake_o), step=epoch)

            if (epoch % save_every) == 0:

                tf.print('Iteration:', epoch, output_stream=sys.stdout)

                recon_imgs_final = fc_inverse_radon.predict(gen_prediction)

                current_sinogram, _ = generator.predict(inpainted_sinogram_masked)
                save2img(recon_imgs_final[0,:,:,0], '%s/recon_it_%05d.png' % (itr_out_dir, epoch))
                save2img(current_sinogram[0,:,:,0], '%s/sinogram_%05d.png' % (itr_out_dir, epoch))

            sys.stdout.flush()
            epoch += 1

        recon_imgs_final = fc_inverse_radon.predict(gen_prediction)
        current_sinogram, _ = generator.predict(inpainted_sinogram_masked)
        save2img(recon_imgs_final[0,:,:,0], '%s/final_recon_%05d.png' % (final_recon_out_dir, idx))
        save2img(current_sinogram[0,:,:,0], '%s/sinogram_%05d.png' % (final_recon_out_dir, idx))
        save2img(inpainted_sinogram_masked.numpy()[0,:,:,0], '%s/original_sinogram_%05d.png' % (final_recon_out_dir, idx))



