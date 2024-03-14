""" @Author: Ashequr Rahman
    @Date: 2022-05-22  
    @Last Modified by: Ashequr Rahman 
"""

from tensorflow.python.framework.ops import disable_eager_execution
import gc
from tensorflow.keras import layers
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input, Conv3D, Conv2D, Conv3DTranspose,UpSampling3D,UpSampling2D,MaxPooling3D, Dropout, BatchNormalization, concatenate, Add, Activation, LeakyReLU
from tensorflow.keras.activations import softmax
from tensorflow.keras.initializers import Constant
from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.callbacks import Callback,ReduceLROnPlateau
import numpy as np
from tensorflow.keras import backend as K
tf.compat.v1.enable_eager_execution()
NORMALIZER = 1e1

def conv_tr_bnorm_relu_drp(nb_filters, krx, kry, krz, name, strd=(2,2,2), bias_ct=0.03, leaky_alpha=0.01, drop_prob=0.0):
  def f(input):
    conv = Conv3DTranspose(  nb_filters, 
                             kernel_size = (krx, kry, krz), # num. of filters and kernel size 
                             strides=strd,
                             padding='same',
                             use_bias=True,
                             kernel_initializer='glorot_normal', 
                             bias_initializer=Constant(value=bias_ct))(input)
    conv = BatchNormalization(name='BN-'+name)(conv)
    conv = LeakyReLU(alpha=leaky_alpha)(conv) # activation func. 

    return conv
  return f


def conv_relu_drp(nb_filters, krx, kry, krz, name, bias_ct=0.03, leaky_alpha=0.01, drop_prob=0.1):
  def f(input):
    conv = Conv3D( nb_filters, 
                   kernel_size = (krx, kry, krz), # num. of filters and kernel size 
                   strides=(1,1,1),
                   padding='same',
                   use_bias=True,
                   kernel_initializer='glorot_normal', 
                   bias_initializer=Constant(value=bias_ct))(input)
    conv = BatchNormalization(name='BN-'+name)(conv)
    conv = LeakyReLU(alpha=leaky_alpha)(conv) # activation func. 
    conv = Dropout(drop_prob)(conv) 

    return conv
  return f

class SlicingLayer(layers.Layer):
  def __init__(self, start_slice = [0, 0, 0, 24, 0], slice_width = [-1, -1, -1, 3, -1], name='Slice_model'): #change: 2/4/23
    super(SlicingLayer, self).__init__(name=name)
    self.start_slice = tf.Variable(initial_value=start_slice, trainable=False, name='start_slice')
    self.slice_width = tf.Variable(initial_value=slice_width, trainable=False, name='slice_width')

  #@tf.function
  def call(self, inputs):
    return tf.slice(inputs, self.start_slice, self.slice_width)

class ChannelShiftLayer(layers.Layer):
  def __init__(self, U, name = 'Channel_shift_model'):
    super(ChannelShiftLayer, self).__init__(name=name)
    self.U = tf.Variable(initial_value=U, trainable=False, name='Channels')
  #@tf.function
  def call(self, loc_img):
    loc_img = UpSampling2D(size=(2,2))(loc_img)
    conv = tf.nn.conv2d(
              loc_img,
              self.U,
              strides=[1, 1, 1, 1],
              padding='SAME'
            )
    conv = tf.expand_dims(conv, -2)
    return conv

class ChannelizeLayer(layers.Layer):
  def __init__(self, name = 'Channelize_model'):
    super(ChannelizeLayer, self).__init__(name=name)

  #@tf.function
  def call(self, f, U_shifted):
    f = UpSampling3D(size=(2,2,1))(f)
    v = tf.divide(tf.reduce_sum(tf.multiply(f, U_shifted), [1, 2]), NORMALIZER)
    return v

def CNN(input,num_reg,names):
  Feat1 = conv_relu_drp(16,3,3,3,names+'F2',drop_prob=0)(input)
  x = MaxPooling3D(pool_size=(2,2,2),name=names+'Pool1')(Feat1)

  Feat2 = conv_relu_drp(32,3,3,3,names+'F4',drop_prob=0)(x)
  x = MaxPooling3D(pool_size=(2,2,2),name=names+'Pool2')(Feat2)

  Feat3 = conv_relu_drp(64,3,3,3,names+'F7',drop_prob=0.0)(x)
  x = MaxPooling3D(pool_size=(2,2,2),name=names+'Pool3')(Feat3)
  

  x = conv_relu_drp(128,3,3,3,names+'Fmid',drop_prob=0.1)(x)
    

  x = conv_tr_bnorm_relu_drp(64,3,3,3,names+'Up1')(x)
  x = Add()([x, Feat3])
  x = conv_relu_drp(64,3,3,3,names+'F8',drop_prob=0.0)(x)

  x = conv_tr_bnorm_relu_drp(32,3,3,3,names+'Up2')(x)
  x = Add()([x, Feat2])
  x = conv_relu_drp(32,3,3,3,names+'F12',drop_prob=0)(x)

  x = conv_tr_bnorm_relu_drp(16,3,3,3,names+'Up3')(x)
  x = Add()([x, Feat1])
  x = conv_relu_drp(16,3,3,3,names+'F13',drop_prob=0)(x)


  x = Conv3D( num_reg, 
                 kernel_size = (1, 1, 1), # num. of filters and kernel size 
                 strides=(1,1,1),
                 padding='same',
                 use_bias=True,
                 kernel_initializer='glorot_normal', 
                 bias_initializer=Constant(value=0.03))(x)

  x = LeakyReLU(alpha=0.00)(x)
  return x


def build_dennet3D(input_shape, loc_shape, num_reg, lambda_val_chdiff, U):
  F_hat_LD = Input(shape = (input_shape), name = 'f_hat_ld')
  F_hat_ND = Input(shape = (input_shape), name = 'f_hat_nd')
  Loc_Img = Input(shape = (loc_shape), name = 'loc_img')

  # build denoise net
  denoise_net_out = CNN(F_hat_LD, num_reg, 'CNN3D')
  denoise_net = Model(
                    inputs = F_hat_LD, 
                    outputs = denoise_net_out
                  )

  # build the full network 
  print(f'Building  the full net ...')
  print(f'Denoiser ...')
  F_hat_ND_pred = denoise_net(F_hat_LD)
  print(f'Channel Shift ...')
  Channels_Shifted = ChannelShiftLayer(U=U, name='ChSh1')(Loc_Img)
  print(f'Slicing ...')
  F_hat_ND_sliced = SlicingLayer(name='Slice1')(F_hat_ND)
  F_hat_ND_pred_sliced = SlicingLayer(name='Slice2')(F_hat_ND_pred)
  print(f'Channelize ...')
  v_ND = ChannelizeLayer(name='chize1')(F_hat_ND_sliced, Channels_Shifted)
  v_ND_pred = ChannelizeLayer(name='chize2')(F_hat_ND_pred_sliced, Channels_Shifted)

  denoise_obs_net = Model(
                      inputs = [F_hat_LD, F_hat_ND, Loc_Img], 
                      outputs = [F_hat_ND_pred, Channels_Shifted, v_ND, v_ND_pred]
                    )

  #========================================
  # Custom Losses
  #========================================
  #========================================
  # Loss1: mse loss of denoising
  #========================================
  denoise_obs_net.add_loss(tf.reduce_mean(tf.math.squared_difference(F_hat_ND, F_hat_ND_pred)))
   
  #========================================
  # Loss2: (channelized difference)
  #========================================
  denoise_obs_net.add_loss(lambda_val_chdiff*tf.reduce_mean(tf.math.squared_difference(v_ND, v_ND_pred)))

  #========================================
  # Custom Metrics
  #========================================
  denoise_obs_net.add_metric(tf.reduce_mean(tf.math.squared_difference(F_hat_ND, F_hat_ND_pred)), 
                              name = 'mse', aggregation = 'mean')
  denoise_obs_net.add_metric(tf.reduce_mean(tf.math.squared_difference(v_ND, v_ND_pred)), 
                              name = 'chv_mse', aggregation = 'mean')

  return denoise_obs_net

def build_dennet3D_pred(input_shape, num_reg):
  F_hat_LD = Input(shape = (input_shape), name = 'f_hat_ld')

  # build denoise net
  denoise_net_out = CNN(F_hat_LD, num_reg, 'CNN3D')
  denoise_net = Model(
                    inputs = F_hat_LD, 
                    outputs = denoise_net_out
                  )

  # build the full network 
  print(f'Building  the full net ...')
  print(f'Denoiser ...')
  F_hat_ND_pred = denoise_net(F_hat_LD)

  denoise_obs_net = Model(
                      inputs = [F_hat_LD], 
                      outputs = [F_hat_ND_pred]
                    )
  return denoise_obs_net

class LossHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []
  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))

class ClearMemory(Callback):
  def on_epoch_end(self, epoch, logs=None):
    gc.collect()
    K.clear_session()

