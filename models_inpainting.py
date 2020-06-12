import tensorflow as tf 
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Conv1D, BatchNormalization, LayerNormalization,\
    Activation, Input, UpSampling2D, MaxPooling2D, MaxPooling1D, SpatialDropout2D, Lambda
import numpy as np 
from tensorflow.keras import layers

import os
import math
from typing import List
import sys
import pydot as pydot

import graphviz

from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file, get_source_inputs

from tensorflow.keras.applications.imagenet_utils import preprocess_input as _preprocess
import tensorflow_addons as tfa

from sn import SpectralNormalization


# Original discriminator
def tomogan_disc(input_shape):
    _tmp = inputs = Input(shape=input_shape)

    _tmp = Conv2D(filters=32, kernel_size=3, padding='same', strides=(2,2),\
                  activation=None)(_tmp)
    _tmp = layers.LeakyReLU(alpha=0.2)(_tmp)
    
    _tmp = Conv2D(filters=32, kernel_size=3, padding='same', \
                  activation=None)(_tmp)

    _tmp = layers.LeakyReLU(alpha=0.2)(_tmp)

    _tmp = Conv2D(filters=64, kernel_size=3, padding='same', strides=(2,2),\
                  activation=None)(_tmp)

    _tmp = layers.LeakyReLU(alpha=0.2)(_tmp)

    _tmp = Conv2D(filters=64, kernel_size=3, padding='same', \
                  activation=None)(_tmp)

    _tmp = layers.LeakyReLU(alpha=0.2)(_tmp)

    _tmp = Conv2D(filters=96, kernel_size=3, padding='same', strides=(2,2),\
                  activation=None)(_tmp)

    _tmp = layers.LeakyReLU(alpha=0.2)(_tmp)

    _tmp = Conv2D(filters=96, kernel_size=3, padding='same', \
                  activation=None)(_tmp)

    _tmp = layers.LeakyReLU(alpha=0.2)(_tmp)

    _tmp = Conv2D(filters=128, kernel_size=3, padding='same', strides=(2,2),\
                  activation=None)(_tmp)

    _tmp = layers.LeakyReLU(alpha=0.2)(_tmp)

    _tmp = Conv2D(filters=128, kernel_size=3, padding='same', \
                  activation=None)(_tmp)

    _tmp = layers.LeakyReLU(alpha=0.2)(_tmp)

    _tmp = Conv2D(filters=160, kernel_size=3, padding='same', strides=(2,2),\
                  activation=None)(_tmp)
   
    _tmp = layers.LeakyReLU(alpha=0.2)(_tmp)

    _tmp = Conv2D(filters=160, kernel_size=3, padding='same', \
                  activation=None)(_tmp)

    _tmp = layers.LeakyReLU(alpha=0.2)(_tmp)

    _tmp = Conv2D(filters=192, kernel_size=3, padding='same', strides=(2,2),\
                  activation=None)(_tmp)
  
    _tmp = layers.LeakyReLU(alpha=0.2)(_tmp)

    _tmp = Conv2D(filters=16, kernel_size=3, padding='same', \
                  activation=None)(_tmp)

    _tmp = layers.LeakyReLU(alpha=0.2)(_tmp)

    _tmp = Conv2D(filters=1, kernel_size=1, padding='same', \
                  activation=None)(_tmp)

    _tmp = layers.Flatten()(_tmp)
    _tmp = layers.Dense(units=32, activation=None)(_tmp)
    _tmp = layers.Dense(units=1, activation="linear")(_tmp)
    
    return tf.keras.models.Model(inputs, _tmp)

def unet_conv_block(inputs, nch, dilation_rate):
    _tmp = Conv2D(filters=nch, kernel_size=3, dilation_rate = dilation_rate, padding='same', activation=None, use_bias = True)(inputs)
    _tmp = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(_tmp)
    _tmp = layers.LeakyReLU(alpha=0.2)(_tmp)

    _tmp = Conv2D(filters=nch, kernel_size=3, dilation_rate = dilation_rate, padding='same', activation=None, use_bias = True)(_tmp)
    _tmp = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(_tmp)
    _tmp = layers.LeakyReLU(alpha=0.2)(_tmp)

    return _tmp


def recon_generator_model(input_shape, n_layers_unet, dilation_rate, use_cnnt = False):
    inputs = Input(shape=input_shape)
    label2idx = {'input': 0}
    
    _tmp = Conv2D(filters=48, kernel_size=17, dilation_rate = dilation_rate, padding='same', activation=None, use_bias = True)(inputs)
    _tmp = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(_tmp)

    _tmp = inputs_concat = layers.LeakyReLU(alpha=0.2)(_tmp)
    ly_outs = [_tmp, ]

    _tmp = unet_conv_block(ly_outs[-1], 48, dilation_rate)
    ly_outs.append(_tmp)
    label2idx['box1_out'] = len(ly_outs)-1

    for ly, nch in zip(range(2, n_layers_unet+1), (64, 96, 192, 384, 512, 512, 512, 512)):
        _tmp = Conv2D(filters=nch, kernel_size=3, strides=2, \
                padding='same', activation=None, use_bias = True)(ly_outs[-1])   

        _tmp = tfa.layers.InstanceNormalization(axis=3, 
                                center=True, 
                                scale=True,
                                beta_initializer="random_uniform",
                                gamma_initializer="random_uniform")(_tmp)  

        _tmp = layers.LeakyReLU(alpha=0.2)(_tmp)  

        _tmp = unet_conv_block(_tmp, nch, dilation_rate)

        ly_outs.append(_tmp)
        label2idx['box%d_out' % (ly)] = len(ly_outs)-1
        
    # intermediate layers
    _tmp = Conv2D(filters=ly_outs[-1].shape[-1], kernel_size=3, strides=2, \
            padding='same', activation=None, use_bias = True)(ly_outs[-1])  

    _tmp = tfa.layers.InstanceNormalization(axis=3, 
                            center=True, 
                            scale=True,
                            beta_initializer="random_uniform",
                            gamma_initializer="random_uniform")(_tmp)  

    _tmp = intermediate_layer = layers.LeakyReLU(alpha=0.2)(_tmp)  

    ly_outs.append(_tmp)
    
    for ly, nch in zip(range(1, n_layers_unet+1), (512, 512, 512, 384, 192, 96, 64, 48, 32)):

        if use_cnnt:
            _tmp = Conv2DTranspose(filters=ly_outs[-1].shape[-1], activation=None, \
                        kernel_size=4, strides=(2, 2), padding='same')(ly_outs[-1])
            _tmp = layers.LeakyReLU(alpha=0.2)(_tmp) 
        else: 
            _tmp = UpSampling2D(size=(2, 2), interpolation='bilinear')(ly_outs[-1]) 
        _tmp = layers.concatenate([ly_outs[label2idx['box%d_out' % (n_layers_unet-ly+1)]], _tmp])

        _tmp = unet_conv_block(_tmp, nch, dilation_rate)
        ly_outs.append(_tmp)


    _tmp = current_sinogram = Conv2D(filters=1, kernel_size=1, padding='same', 
                  activation=None, use_bias = True)(_tmp)

    return tf.keras.models.Model(inputs, [_tmp, intermediate_layer])

def fc_model(input_shape, img_size):
    inputs = Input(shape=input_shape)
    _tmp = layers.Flatten()(inputs)

    _tmp = layers.Dense(units=64, activation='tanh', use_bias = True)(_tmp)

    _tmp = layers.Dense(units=img_size**2, activation=None, use_bias = True)(_tmp)

    outputs = tf.reshape(_tmp, shape=(1,img_size,img_size,1))
    
    return tf.keras.models.Model(inputs, outputs)


if __name__ == '__main__':

    disc_model = tomogan_disc((1024, 1024,3))
    disc_model.summary()

    recon_generator_model = recon_generator_model((800, 1024,3), n_layers_unet=5)
    recon_generator_model.summary()

    tf.keras.utils.plot_model(
        disc_model, to_file='./disc_model.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=True, dpi=96)
        
    tf.keras.utils.plot_model(
        recon_generator_model, to_file='./recon_generator.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=True, dpi=96)


