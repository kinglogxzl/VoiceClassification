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
=============各类别准确率=============
类别:  21       样本量:  389    准确率:  0.9049         类别名: 自助转账=自助转账
类别:  168      样本量:  80     准确率:  0.95   类别名: 学费缴纳=学费缴纳|交学费|我要交学费
类别:  3        样本量:  82     准确率:  0.9024         类别名: 账户余额=账户余额|余额查询|查余额|我有多少钱
类别:  2        样本量:  47     准确率:  0.8511         类别名: 交易明细=交易明细|交易记录
类别:  1        样本量:  41     准确率:  0.7317         类别名: 我的账户=我的账户
类别:  47       样本量:  35     准确率:  0.8286         类别名: 跨行转账=跨行转账|我要转账|我要转钱|我要汇款|我要打钱
类别:  45       样本量:  36     准确率:  0.5278         类别名: 收款人管理=收款人|登记簿|收款人管理|登记簿管理
类别:  102      样本量:  24     准确率:  0.7917         类别名: 定活互转=定活互转|定转活|活转定
类别:  22       样本量:  23     准确率:  0.6957         类别名: 转账查询=转账查询
类别:  104      样本量:  20     准确率:  0.8    类别名: 益农存=益农存|一农村|艺农村|艺侬存|你农村|地农村
类别:  4        样本量:  35     准确率:  0.4    类别名: 取款密码修改=取款密码修改
类别:  234      样本量:  18     准确率:  0.7222         类别名: 绑定动态口令=动态口令|绑定动态口令
类别:  208      样本量:  13     准确率:  0.8462         类别名: 安全中心=安全中心
类别:  147      样本量:  16     准确率:  0.625  类别名: 社会保险=社会保险
类别:  103      样本量:  17     准确率:  0.5294         类别名: 通知存款=通知存款
类别:  80       样本量:  12     准确率:  0.5    类别名: 信用卡还款=信用卡还款
类别:  49       样本量:  9      准确率:  0.6667         类别名: 我的贷款=我的贷款
类别:  247      样本量:  8      准确率:  0.75   类别名: 版本信息=版本信息
类别:  51       样本量:  7      准确率:  0.7143         类别名: 还钱=还钱|还款|我要还款|我要还钱
类别:  105      样本量:  6      准确率:  0.6667         类别名: 益农存转入转出=益农存转入|益农存转出|益农存转入转出|一农村转入|一农村转出|一农村转入转出|艺农村转入|艺农村转出|艺农村转入转出|艺侬存转入|艺侬存转出|艺侬存转入转出|你农村转入|你农村转出|你农村转入转出|地农村转入|地农村转出|地农村转入转出|农村转入|农村转出|农村转入转出
类别:  216      样本量:  15     准确率:  0.2667         类别名: 登录密码修改=登录密码修改
类别:  211      样本量:  13     准确率:  0.3077         类别名: 短信签约=短信签约
类别:  146      样本量:  11     准确率:  0.3636         类别名: 生活缴费=生活缴费|生活交费|交电费|缴电费|交水费|缴水费|交煤气费|交燃气费|缴煤气费|缴燃气费
类别:  164      样本量:  9      准确率:  0.3333         类别名: ETC线上申请=etc申请|etc线上申请
类别:  76       样本量:  8      准确率:  0.375  类别名: 信用卡=信用卡
类别:  0        样本量:  7      准确率:  0.4286         类别名: 账户管理=账户管理
类别:  78       样本量:  18     准确率:  0.1667         类别名: 我的账单=信用卡账单|我的账单
类别:  48       样本量:  5      准确率:  0.6    类别名: 贷款服务=贷款服务
类别:  245      样本量:  6      准确率:  0.5    类别名: 联系我们=联系我们
类别:  73       样本量:  7      准确率:  0.4286         类别名: 银行网点=银行网点
类别:  185      样本量:  5      准确率:  0.6    类别名: 美团外卖=叫外卖|我要叫外卖|美团外卖
类别:  226      样本量:  5      准确率:  0.6    类别名: 交易安全锁=交易安全锁|交易安全所
类别:  190      样本量:  6      准确率:  0.3333         类别名: 话费=话费|充话费|充手机费
类别:  20       样本量:  4      准确率:  0.5    类别名: 转账支付=转账支付
类别:  209      样本量:  4      准确率:  0.5    类别名: 预留信息=预留信息|预留信息管理
类别:  9        样本量:  6      准确率:  0.3333         类别名: 无卡取现=无卡取现
类别:  219      样本量:  4      准确率:  0.5    类别名: 设备管理=设备管理
类别:  160      样本量:  4      准确率:  0.5    类别名: 车生活=车生活
类别:  14       样本量:  3      准确率:  0.6667         类别名: 他行账户管理=他行账户管理
类别:  117      样本量:  3      准确率:  0.6667         类别名: 聚鑫宝=聚鑫宝
类别:  57       样本量:  4      准确率:  0.5    类别名: 商户管理=商户管理
类别:  24       样本量:  4      准确率:  0.25   类别名: 手机号转账=手机号转账
类别:  46       样本量:  12     准确率:  0.0833         类别名: 行内转账=行内转账|我要转账|我要转钱|我要汇款|我要打钱
类别:  52       样本量:  6      准确率:  0.1667         类别名: 贷款查询=贷款查询
类别:  101      样本量:  2      准确率:  0.5    类别名: 储蓄服务=储蓄服务
类别:  213      样本量:  9      准确率:  0.1111         类别名: 指纹登录=指纹登录|指纹登陆|指纹管理
类别:  217      样本量:  3      准确率:  0.3333         类别名: 手势密码=手势密码
'''