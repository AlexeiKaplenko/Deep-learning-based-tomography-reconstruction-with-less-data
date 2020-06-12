import numpy as np 
import h5py, threading
import queue as Queue
import h5py, glob
from util import scale2uint8, save2img
import sys
import os
import cv2
import tensorflow as tf


class bkgdGen(threading.Thread):
    def __init__(self, data_generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = data_generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            # block if necessary until a free slot is available
            self.queue.put(item, block=True, timeout=None)
        self.queue.put(None)

    def next(self):
        # block if necessary until an item is available
        next_item = self.queue.get(block=True, timeout=None)
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

def gen_train_batch_bg(x_fn, y_fn, mb_size, in_depth, img_size):
    X, Y = None, None
    with h5py.File(x_fn, 'r') as hdf_fd:
        X = hdf_fd['images'][:].astype(np.float32)

    with h5py.File(y_fn, 'r') as hdf_fd:
        Y = hdf_fd['images'][:].astype(np.float32)

    while True:
        idx = np.random.randint(0, X.shape[0]-in_depth, mb_size)
        crop_idx = np.random.randint(0, X.shape[1]-img_size)
        
        batch_X = np.array([np.transpose(X[s_idx : (s_idx+in_depth)], (1, 2, 0)) for s_idx in idx])
        batch_X = batch_X[:, crop_idx:(crop_idx+img_size), crop_idx:(crop_idx+img_size), :]

        batch_Y = np.expand_dims([Y[s_idx+in_depth//2] for s_idx in idx], 3)
        batch_Y = batch_Y[:, crop_idx:(crop_idx+img_size), crop_idx:(crop_idx+img_size), :]

        batch_X = (batch_X - 127.5)/127.5
        batch_Y = (batch_Y - 127.5)/127.5


        # yield 2*batch_X/255-1, 2*batch_Y/255-1 #normalization to [-1,1]
        # yield batch_X/255, batch_Y/255
        yield batch_X, batch_Y


def get_train_slice(x_fn, in_depth, img_size):
    X = h5py.File(x_fn, 'r')['images']
    # Y = h5py.File(y_fn, 'r')['images']

    idx = X.shape[0]//2

    height, width = X.shape[1], X.shape[2]

    batch_X = X[idx-in_depth : idx+1+in_depth, (height-img_size)//2:(height+img_size)//2, (width-img_size)//2:(width+img_size)//2]


    batch_X = (batch_X - 127.5)/127.5

    return batch_X.astype(np.float32) 

def prepare_training_array(x_fn, img_size):
    X = h5py.File(x_fn, 'r')['images']
    height, width = X.shape[1], X.shape[2]
    X = X[:, (height-img_size)//2:(height+img_size)//2, (width-img_size)//2:(width+img_size)//2]
    X = (X - 127.5)/127.5
    return X.astype(np.float32) 


@tf.function 
def normalize_tensor(x): 
    return ((x - tf.reduce_min(x)) / (tf.reduce_max - tf.reduce_min) -0.5) / 0.5 

def get1batch4test(x_fn, in_depth, img_size):
    X = h5py.File(x_fn, 'r')['images']

    idx = X.shape[0]//2

    X = np.transpose(X[idx : idx+in_depth], (1, 2, 0))
    batch_X = np.expand_dims(X, 0)

    tf.print('X.shape', batch_X.shape, output_stream=sys.stdout)

    batch_X = (batch_X - 127.5)/127.5

    return batch_X.astype(np.float32) 
    
def inference_stack(x_fn, in_depth, generator, output_folder, experiment_name):
    X = h5py.File(x_fn, 'r')['images']
    stack_z_len = X.shape[0]

    for s_idx in range(stack_z_len-in_depth):
        batch_X = np.expand_dims(np.transpose(X[s_idx : (s_idx+in_depth)], (1, 2, 0)), axis=0)
        batch_X.astype(np.float32) 

        pred_img = generator.predict(batch_X)

        output = output_folder+"/"+experiment_name

        if not os.path.exists(output):
            os.makedirs(output)

        save2img(pred_img[0,:,:,0], output+"/"+'%s.png' % (s_idx))

def get_sinogram(x_fn, img_size, in_depth=None):

    with h5py.File(x_fn, 'r') as hdf_fd:
        X = hdf_fd['projs'][:].astype(np.float32)

    empty_array = np.zeros((X.shape[0], img_size))

    X = X[:,500]

    for i in range(X.shape[0]):
        empty_array[i] = X[i][500]

    batch_X = np.expand_dims(X, axis=0)

    batch_X = np.expand_dims(batch_X, axis=-1)

    batch_X = (batch_X - 127.5)/127.5

    batch_X = batch_X[:,:,384:896,:]

    return batch_X




