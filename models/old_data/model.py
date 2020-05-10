from __future__ import print_function

from keras import backend as K
import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation, Reshape, Permute
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.layers.advanced_activations import ELU
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.layers import Reshape

from os.path import isfile

from network.muti_gpu import *
from tensorflow.python.client import device_lib
from network.muti_gpu import make_parallel, get_available_gpus
import h5py
def MyCNN_Keras2(X_shape, nb_classes, nb_layers=4, reshape_x=39):
    # Inputs:
    #    X_shape = [ # spectrograms per batch, # audio channels, # spectrogram freq bins, # spectrogram time bins ]
    #    nb_classes = number of output n_classes
    #    nb_layers = number of conv-pooling sets in the CNN
    from keras import backend as K
    K.set_image_data_format('channels_last')                   # SHH changed on 3/1/2018 b/c tensorflow prefers channels_last

    nb_filters = 32  # number of convolutional filters = "feature maps"
    # nb_filters2 = 16
    kernel_size = (3, 3)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for max pooling
    cl_dropout = 0.4    # conv. layer dropout
    dl_dropout = 0.4   # dense layer dropout

    print("MyCNN_Keras2: X_shape = ",X_shape,", channels = ",X_shape[3])
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size, W_regularizer=l2(0.01), input_shape=input_shape,padding='same', name="Input"))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Activation('relu'))        # Leave this relu & BN here.  ELU is not good here (my experience)
    # model.add(BatchNormalization(axis=-1))  # axis=1 for 'channels_first'; but tensorflow preferse channels_last (axis=-1)

    for layer in range(nb_layers-1):   # add more layers than just the first
        nb_filters = nb_filters*2
        model.add(Conv2D(nb_filters, kernel_size,W_regularizer=l2(0.01), padding='same'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Activation('elu'))
        model.add(Dropout(cl_dropout))
        #model.add(BatchNormalization(axis=-1))  # ELU authors reccommend no BatchNorm. I confirm.

    model.add(Permute((2, 1, 3)))
    model.add(Reshape((reshape_x, -1)))
    #model.add(Reshape((39, -1)))
    model.add(LSTM(50, return_sequences=True,dropout=0.3,recurrent_dropout=0.3))
    model.add(Flatten())
    model.add(Dense(256))            # 128 is 'arbitrary' for now
    #model.add(Activation('relu'))   # relu (no BN) works ok here, however ELU works a bit better...
    model.add(Activation('elu'))
    model.add(Dropout(dl_dropout))
    model.add(Dense(nb_classes,W_regularizer=l2(0.01)))
    model.add(Activation("softmax",name="Output"))
    return model

'''
训练记录：
1.先将dropout、正则系数全部取消，进行训练
2.逐步增加dropout、正则降低过拟合
model说明:
oldd_weights_6166.hdf5 
train_data: 0.8+
val_data: 0.8+
test_data: 0.6166
'''