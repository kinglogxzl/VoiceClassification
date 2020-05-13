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

from keras.layers import Reshape

from keras.regularizers import l2

from os.path import isfile

from network.muti_gpu import *
from tensorflow.python.client import device_lib
from network.muti_gpu import make_parallel, get_available_gpus
import h5py


# This is a VGG-style network that I made by 'dumbing down' @keunwoochoi's compact_cnn code
# I have not attempted much optimization, however it *is* fairly understandable
def MyCNN_Keras2(X_shape, nb_classes, nb_layers=4, reshape_x=39, drop_out_arg=[]):
    # Inputs:
    #    X_shape = [ # spectrograms per batch, # audio channels, # spectrogram freq bins, # spectrogram time bins ]
    #    nb_classes = number of output n_classes
    #    nb_layers = number of conv-pooling sets in the CNN
    from keras import backend as K
    K.set_image_data_format('channels_last')  # SHH changed on 3/1/2018 b/c tensorflow prefers channels_last

    if (drop_out_arg == []):
        drop_out_arg = [0.4, 0.4, 0.3, 0.3, 0.01]

    nb_filters = 32  # number of convolutional filters = "feature maps"
    # nb_filters2 = 16
    kernel_size = (3, 3)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for max pooling
    cl_dropout = drop_out_arg[0]  # conv. layer dropout
    dl_dropout = drop_out_arg[1]  # dense layer dropout

    print("MyCNN_Keras2: X_shape = ", X_shape, ", channels = ", X_shape[3])
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    model = Sequential()
    model.add(
        Conv2D(nb_filters, kernel_size, W_regularizer=l2(drop_out_arg[4]), input_shape=input_shape, padding='same',
               name="Input"))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Activation('relu'))  # Leave this relu & BN here.  ELU is not good here (my experience)
    # model.add(BatchNormalization(axis=-1))  # axis=1 for 'channels_first'; but tensorflow preferse channels_last (axis=-1)

    for layer in range(nb_layers - 1):  # add more layers than just the first
        nb_filters = nb_filters * 2
        model.add(Conv2D(nb_filters, kernel_size, padding='same', W_regularizer=l2(drop_out_arg[4])))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Activation('elu'))
        model.add(Dropout(cl_dropout))
        # model.add(BatchNormalization(axis=-1))  # ELU authors reccommend no BatchNorm. I confirm.

    model.add(Permute((2, 1, 3)))
    model.add(Reshape((reshape_x, -1)))
    # model.add(Reshape((39, -1)))
    model.add(LSTM(40, return_sequences=True, dropout=drop_out_arg[2], recurrent_dropout=drop_out_arg[3]))
    model.add(Flatten())
    model.add(Dense(256))  # 128 is 'arbitrary' for now
    # model.add(Activation('relu'))   # relu (no BN) works ok here, however ELU works a bit better...
    model.add(Activation('elu'))
    # model.add(Activation('tanh'))
    model.add(Dropout(dl_dropout))
    model.add(Dense(nb_classes, W_regularizer=l2(drop_out_arg[4])))
    model.add(Activation("softmax", name="Output"))
    return model

'''
实验记录：
1.把lstm层的输出维度改为40
2.dropout和正则系数依次为（具体所指见上面代码）
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0.2, 0.2, 0.1, 0.1, 0],
 [0.4, 0.4, 0.3, 0.3, 0.0],
 [0.4, 0.4, 0.3, 0.3, 0.01]],epochs均为100
3.train_loss降低而val_loss不降低时，不使用验证集(--val=0)，用[0.4, 0.4, 0.3, 0.3, 0.01]来训练
4.test_acc再次难以提升后，用[0, 0, 0, 0, 0]训练train_acc到90+（大约8个epoch）
5.得到需要关注的类别后，增加相应类别的计算loss的权重
6.从第3步之后得到的参数开始，（不用验证集）用[0.2, 0.2, 0.1, 0.1, 0]训练到过拟合。

'''