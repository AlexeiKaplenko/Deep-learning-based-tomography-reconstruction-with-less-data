#! /home/zliu0/usr/anaconda3/envs/tfgpu/bin/python
import tensorflow as tf 
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from functools import partial

import numpy as np 
import os
from util import save2img
import sys, os, time, argparse, shutil, scipy, h5py, glob
from models_inpainting import tomogan_disc as make_discriminator_model, recon_generator_model  # import a disc model
from data_processor import bkgdGen, gen_train_batch_bg, get1batch4test, get_train_slice, prepare_training_array
import datetime
from sn import SpectralNormalization

from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.layers.projection_2d import parallel_projection2d
from pyronn.ct_reconstruction.layers.backprojection_2d import parallel_backprojection2d

parser = argparse.ArgumentParser(description='encode sinogram image.')
parser.add_argument('-gpus',  type=str, default="0", help='list of visiable GPUs')
parser.add_argument('-expName', type=str, default="Reconstruction", required=False, help='Experiment name')
parser.add_argument('-lpixel', type=float, default=1., help='lambda pixel loss')
parser.add_argument('-ladv', type=float, default=4e-4, help='lambda adv') 
parser.add_argument('-lunet', type=int, default=9, help='Unet layers')
parser.add_argument('-depth', type=int, default=3, help='input number of channels')
parser.add_argument('-itg', type=int, default=1, help='iterations for G')
parser.add_argument('-itd', type=int, default=1, help='iterations for D')
parser.add_argument('-n_proj', type=int, default=256, help='number of projections taken for sparse sinogram')
parser.add_argument('-n_proj_inpaint', type=int, default=1024, help='number of projections for inpainted sinogram')
parser.add_argument('-tfolder', type=str, default='./train_dataset/', required=True, help='path to train dataset folder')

args, unparsed = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR


@tf.function 
def normalize_tensor(x): 
    return ((x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x)) -0.5) / 0.5 

number_epochs = 100
mb_size = 4
img_size = 512
epsilon = 1e-14
in_depth = 1

# Volume Parameters:
volume_size = img_size
volume_shape = [volume_size, volume_size]
volume_spacing = [1, 1]

# Detector Parameters:
detector_shape = img_size
detector_spacing = 1


# Trajectory Parameters:
number_of_projections = args.n_proj # 256, must be devisible by 2**args.lunet number of U-Nat layers
angular_range = np.pi 

number_of_projections_inpainted = args.n_proj_inpaint # 1024, must be devisible by 2**args.lunet number of U-Nat layers

# angular_range_inpainted = np.radians(270)
angular_range_inpainted = np.pi

disc_iters, gene_iters = args.itd, args.itg
lambda_pixel, lambda_adv = args.lpixel, args.ladv

it_per_slice = 1
save_every = 100

itr_out_dir = args.expName 
if os.path.isdir(itr_out_dir): 
    shutil.rmtree(itr_out_dir)
os.mkdir(itr_out_dir) # to save temp output

# redirect print to a file
sys.stdout = open('%s/%s' % (itr_out_dir, 'iter-prints.log'), 'w') 

#Mask inpainting
np_mask = np.zeros((number_of_projections_inpainted, img_size))
np_mask[0::(number_of_projections_inpainted//number_of_projections)] = 1

# #Mask missing wedge
# np_mask = np.ones((number_of_projections_inpainted, img_size))
# np_mask[:20] = 0
# np_mask[108:] = 0

tf_mask = tf.convert_to_tensor(np_mask, np.float32)

tf_mask = tf.expand_dims(tf_mask, axis = 0)
tf_mask = tf.expand_dims(tf_mask, axis = -1)

generator = recon_generator_model(input_shape=(number_of_projections_inpainted, img_size, 2*in_depth+1), dilation_rate=1, n_layers_unet=args.lunet)

discriminator = make_discriminator_model(input_shape=(number_of_projections_inpainted, img_size, 2*in_depth+1))

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

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_gen_log_dir = './logs/gradient_tape/' + current_time + '/generator_train'


train_dis_log_dir = './logs/gradient_tape/' + current_time + '/discriminator_train'

train_gen_summary_writer = tf.summary.create_file_writer(train_gen_log_dir)

train_dis_summary_writer = tf.summary.create_file_writer(train_dis_log_dir)

geometry_inpainted = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections_inpainted, angular_range_inpainted)
geometry_inpainted.set_trajectory(circular_trajectory.circular_trajectory_2d(geometry_inpainted))

dataset_paths =  [x[2] for x in os.walk('./train_dataset')]
dataset_paths = [args.tfolder + s for s in dataset_paths[0]]
current_it = 0
for epoch in range(number_epochs):

    for dataset_path in dataset_paths:
        #Get and prepare training array
        X = prepare_training_array(dataset_path, img_size)
        for idx in range(X.shape[0]):

            if idx == 0:
                inpainted_sinogram_0 = parallel_projection2d(np.expand_dims(X[0], 0), geometry_inpainted)
            else:
                inpainted_sinogram_0 = parallel_projection2d(np.expand_dims(X[idx-1], 0), geometry_inpainted)

            inpainted_sinogram_0 = tf.keras.layers.LayerNormalization()(inpainted_sinogram_0)
            inpainted_sinogram_0 = tf.expand_dims(inpainted_sinogram_0, axis = -1)

            inpainted_sinogram_1 = parallel_projection2d(np.expand_dims(X[idx], axis=0), geometry_inpainted)
            inpainted_sinogram_1 = tf.keras.layers.LayerNormalization()(inpainted_sinogram_1)
            inpainted_sinogram_1 = tf.expand_dims(inpainted_sinogram_1, axis = -1)


            if idx == (X.shape[0]-1):
                inpainted_sinogram_2 = parallel_projection2d(np.expand_dims(X[idx], axis=0), geometry_inpainted)
            else:
                inpainted_sinogram_2 = parallel_projection2d(np.expand_dims(X[idx+1], axis=0), geometry_inpainted)

            inpainted_sinogram_2 = tf.keras.layers.LayerNormalization()(inpainted_sinogram_2)
            inpainted_sinogram_2 = tf.expand_dims(inpainted_sinogram_2, axis = -1)
            
            inpainted_sinogram = tf.concat([inpainted_sinogram_0, inpainted_sinogram_1, inpainted_sinogram_2], axis = -1)
            inpainted_sinogram_masked = tf_mask * inpainted_sinogram

            for index in range(it_per_slice):
                # current_it = idx*it_per_slice+epoch
                time_git_st = time.time()
                for _ge in range(gene_iters):
                    with tf.GradientTape() as gen_tape:
                        gen_tape.watch(generator.trainable_variables)

                        recon_imgs, _ = generator(inpainted_sinogram_masked, training=True)

                        generated_sinogram_3D = tf.concat([inpainted_sinogram_0, recon_imgs, inpainted_sinogram_2], axis = -1)

                        disc_fake_o = discriminator(generated_sinogram_3D, training=True)
                        disc_real_o = discriminator(inpainted_sinogram, training=True)

                        loss_pixel = tf.keras.losses.MAE(inpainted_sinogram_1, recon_imgs)

                        loss_adv = g_loss_fn(disc_fake_o)

                        gen_loss = lambda_adv * loss_adv + lambda_pixel * loss_pixel 
                        gen_loss = tf.expand_dims(gen_loss, axis = -1)

                    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
                    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

                if current_it < 10:
                    tf.print('gen_elapse: {}s/itr'.format((time.time() - time_git_st)/gene_iters, ), output_stream=sys.stdout)

                with train_gen_summary_writer.as_default():
                    tf.summary.scalar('generator_loss', tf.keras.backend.mean(gen_loss), step=current_it)
                    tf.summary.scalar('generator_loss_pixel', tf.keras.backend.mean(lambda_pixel * loss_pixel), step=current_it)
                    tf.summary.scalar('generator_loss_adv', tf.keras.backend.mean(loss_adv*lambda_adv), step=current_it)

                time_dit_st = time.time()

                for _dis in range(disc_iters):
                    with tf.GradientTape() as disc_tape:
                        disc_tape.watch(discriminator.trainable_variables)

                        disc_fake_o = discriminator(generated_sinogram_3D, training=True)
                        disc_real_o = discriminator(inpainted_sinogram, training=True)

                        gp = gradient_penalty(inpainted_sinogram, generated_sinogram_3D, partial(discriminator, training=True))
                        disc_loss = d_loss_fn(disc_fake_o, disc_real_o) + 10 * gp

                    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

                if current_it < 10:
                    tf.print('disc_elapse: {}s/itr'.format((time.time() - time_dit_st)/disc_iters, ), output_stream=sys.stdout)


                with train_dis_summary_writer.as_default():
                    tf.summary.scalar('discriminator_loss', tf.reduce_mean(disc_loss), step=current_it)
                    tf.summary.scalar('discriminator_loss_real', tf.reduce_mean(disc_real_o), step=current_it)
                    tf.summary.scalar('discriminator_loss_fake', tf.reduce_mean(disc_fake_o), step=current_it)

                if (current_it) % (save_every//gene_iters) == 0:
                    tf.print('current_iteration:', current_it, output_stream=sys.stdout)
                    tf.print('dataset_path:', dataset_path, output_stream=sys.stdout)
                    tf.print('current_slice:', idx, output_stream=sys.stdout)

                    pred_img, current_sinogram = generator.predict(inpainted_sinogram_masked)
                    save2img(pred_img[0,:,:,0], '%s/it%05d.png' % (itr_out_dir, current_it))

                    generator.save("%s/gen-it%05d.h5" % (itr_out_dir, current_it), \
                                include_optimizer=True)

                    discriminator.save("%s/disc-it%05d.h5" % (itr_out_dir, current_it), \
                                include_optimizer=True)

                sys.stdout.flush()
                current_it += 1
