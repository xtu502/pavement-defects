# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 22:23:24 2019

@author: HP
"""

from keras import backend as K
from keras import optimizers
import keras.backend.tensorflow_backend as KTF
import glob
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Activation,ZeroPadding2D
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from keras.layers import Activation, Dense
from matplotlib import pyplot as plt
from skimage import io,data
import time
from keras import layers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

from config import IMAGE_ORDERING

import os,sys
os.getcwd()
os.chdir("/home/cjd/47_road_defect")

print(os.getcwd())
print (sys.version)


#import os
# 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,2"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


def relu6(x):
    return K.relu(x, max_value=6)
# judge whether it is the channels_first or channels_last
channel_axis = 1 if K.image_data_format() == "channels_first" else 3



import tensorflow as tf        #enhanced Focal loss
def focal_loss(gamma=2.):          
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        return -K.sum( K.pow(1. - pt_1, gamma) * K.log(pt_1)) 
    return focal_loss_fixed


def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):  
    if name is not None:  
        bn_name = name + '_bn'  
        conv_name = name + '_conv'  
    else:  
        bn_name = None  
        conv_name = None  
  
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
    x = BatchNormalization(axis=3,name=bn_name)(x)  
    return x  

def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):  
    x = Conv2d_BN(inpt,nb_filter=nb_filter[0],kernel_size=(1,1),strides=strides,padding='same')  
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3,3), padding='same')  
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1,1), padding='same')  
    if with_conv_shortcut:  
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter[2],strides=strides,kernel_size=kernel_size)  
        x = add([x,shortcut])  
        return x  
    else:  
        x = add([x,inpt])  
        return x  


batch_size = 64 
epochs = 30
MODEL_INIT = './obj_reco/init_model.h5'
MODEL_PATH = './obj_reco/tst_model.h5'
board_name1 = './obj_reco/stage1/' + now + '/'
board_name2 = './obj_reco/stage2/' + now + '/'

train_dir='./train_seg/'
validation_dir='./test_seg/'
img_size = (224, 224)  
#classes=list(range(1,5))
#classes=['1','2','3','4']
nb_train_samples = len(glob.glob(train_dir + '/*/*.*'))  
nb_validation_samples = len(glob.glob(validation_dir + '/*/*.*'))  

classes = sorted([o for o in os.listdir(train_dir)])  




# CAM
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])

# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


def cbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])


class SeBlock(keras.layers.Layer):   
    def __init__(self, reduction=4,**kwargs):
        super(SeBlock,self).__init__(**kwargs)
        self.reduction = reduction
    def build(self,input_shape):#
    	#input_shape     
    	pass
    def call(self, inputs):
        x = keras.layers.GlobalAveragePooling2D()(inputs)
        x = keras.layers.Dense(int(x.shape[-1]) // self.reduction, use_bias=False,activation=keras.activations.relu)(x)
        x = keras.layers.Dense(int(inputs.shape[-1]), use_bias=False,activation=keras.activations.hard_sigmoid)(x)
        return keras.layers.Multiply()([inputs,x])    #
        #return inputs*x 



def _conv_block(inputs, filters, kernel_size, alpha, strides=(1, 1)):

    kernel=(kernel_size, kernel_size)
    channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad',
                      data_format=IMAGE_ORDERING)(inputs)
    x = Conv2D(filters, kernel, data_format=IMAGE_ORDERING,
               padding='valid',
               use_bias=False,
               strides=strides,
               name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, kernel_size, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING,
                      name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((kernel_size, kernel_size), data_format=IMAGE_ORDERING,
                        padding='valid',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1), data_format=IMAGE_ORDERING,
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis,
                           name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


#--------------Mobile-ATT-----------------------------------------------------------
#coding=utf-8
from keras.models import *
from keras.layers import *
import keras.backend as K
import keras
import keras.layers as KL
import numpy as np
import tensorflow as tf

alpha = 1.0
depth_multiplier = 1
dropout = 1e-3
 
img_input = Input(shape=(224,224,3))

x = _conv_block(img_input, 32, 3, alpha, strides=(2, 2))
x = _depthwise_conv_block(x, 64, 3, alpha, depth_multiplier, block_id=1)
x = cbam_module(x)


x = _depthwise_conv_block(x, 128, 3, alpha, depth_multiplier,
                          strides=(2, 2), block_id=2)
x = _depthwise_conv_block(x, 128,  3, alpha, depth_multiplier, block_id=3)
x = cbam_module(x)


x = _depthwise_conv_block(x, 256, 3, alpha, depth_multiplier,
                          strides=(2, 2), block_id=4)
x = _depthwise_conv_block(x, 256, 3, alpha, depth_multiplier, block_id=5)
x = cbam_module(x)


x = _depthwise_conv_block(x, 512, 3, alpha, depth_multiplier,
                          strides=(2, 2), block_id=6)
x = _depthwise_conv_block(x, 512, 3, alpha, depth_multiplier, block_id=7)
x = _depthwise_conv_block(x, 512, 3, alpha, depth_multiplier, block_id=8)
x = _depthwise_conv_block(x, 512, 3, alpha, depth_multiplier, block_id=9)
x = _depthwise_conv_block(x, 512, 1, alpha, depth_multiplier, block_id=10)
x = _depthwise_conv_block(x, 512, 5, alpha, depth_multiplier, block_id=11)
x = cbam_module(x)
#    x = SeBlock()(x)
#    x = spatial_attention(x)


x = _depthwise_conv_block(x, 1024, 3, alpha, depth_multiplier,
                          strides=(2, 2), block_id=12)
x = _depthwise_conv_block(x, 1024, 3, alpha, depth_multiplier, block_id=13)
x = cbam_module(x)


x = GlobalAveragePooling2D()(x)
#x = Dense(len(ont_hot_labels[0]),activation='softmax')(x) 
predictions = Dense(len(classes), activation='softmax')(x)
model = Model(inputs=img_input ,outputs=predictions)
#
model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics = ['accuracy'])  #rmsprop




train_datagen = ImageDataGenerator(validation_split=0.2)
train_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))  # 去掉imagenet BGR均值
train_data = train_datagen.flow_from_directory(train_dir, target_size=img_size, classes=classes)
validation_datagen = ImageDataGenerator()
validation_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))
validation_data = validation_datagen.flow_from_directory(validation_dir, target_size=img_size, classes=classes)

#model_checkpoint1 = ModelCheckpoint(filepath=MODEL_INIT, save_best_only=True, monitor='val_accuracy', mode='max')
model_checkpoint1 = ModelCheckpoint(filepath=MODEL_INIT, monitor='val_accuracy')
board1 = TensorBoard(log_dir=board_name1,
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)
callback_list1 = [model_checkpoint1, board1]

model.fit_generator(train_data, steps_per_epoch=nb_train_samples / float(batch_size),
                           epochs = epochs,
                           validation_steps=nb_validation_samples / float(batch_size),
                           validation_data=validation_data,
                           callbacks=callback_list1, verbose=2)

#---------------the 2nd stage---------------------------------------------
model_checkpoint2 = ModelCheckpoint(filepath=MODEL_PATH,  monitor='val_accuracy')
board2 = TensorBoard(log_dir=board_name2,
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)
callback_list2 = [model_checkpoint2, board2]

model.load_weights(MODEL_INIT)
for model1 in model.layers:
    model1.trainable = True

model.compile(optimizer=optimizers.SGD(lr=0.0001), loss = [focal_loss(gamma=2)], metrics=['accuracy']) #loss='categorical_crossentropy',


model.fit_generator(train_data, steps_per_epoch=nb_train_samples / float(batch_size), epochs=epochs,
                    validation_data=validation_data, validation_steps=nb_validation_samples / float(batch_size),
                    callbacks=callback_list2, verbose=2)

'''
#model.summary() saved to file
from contextlib import redirect_stdout   
with open('model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary(line_length=200,positions=[0.30,0.60,0.7,1.0])
'''





