from keras.models import *
from keras.layers import *
import keras.backend as K
import keras
import keras.layers as KL
import numpy as np
import tensorflow as tf


from .config import IMAGE_ORDERING

BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                    'releases/download/v0.6/')


def relu6(x):
    return K.relu(x, max_value=6)



# judge whether it is the channels_first or channels_last
channel_axis = 1 if K.image_data_format() == "channels_first" else 3

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
        return keras.layers.Multiply()([inputs,x])    #weighted channels
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


def get_SAmobilenet_encoder(input_height=224, input_width=224,
                          pretrained='imagenet', channels=3):

    # todo add more alpha and stuff

    assert (K.image_data_format() ==
            'channels_last'), "Currently only channels last mode is supported"
    assert (IMAGE_ORDERING ==
            'channels_last'), "Currently only channels last mode is supported"

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    alpha = 1.0
    depth_multiplier = 1
    dropout = 1e-3

    img_input = Input(shape=(input_height, input_width, channels))

    x = _conv_block(img_input, 32, 3, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, 3, alpha, depth_multiplier, block_id=1)
    x = cbam_module(x)
    f1 = x

    x = _depthwise_conv_block(x, 128, 3, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128,  3, alpha, depth_multiplier, block_id=3)
    x = cbam_module(x)
    f2 = x

    x = _depthwise_conv_block(x, 256, 3, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, 3, alpha, depth_multiplier, block_id=5)
    x = cbam_module(x)
    f3 = x

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
    f4 = x

    x = _depthwise_conv_block(x, 1024, 3, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, 3, alpha, depth_multiplier, block_id=13)
    x = cbam_module(x)
    f5 = x

    if pretrained == 'imagenet':
        model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)

        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = keras.utils.get_file(model_name, weight_path)

        Model(img_input, x).load_weights(weights_path, by_name=True, skip_mismatch=True)

    return img_input, [f1, f2, f3, f4, f5]
