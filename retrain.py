""" @Author: Ashequr Rahman
    @Date: 2022-05-22  
    @Last Modified by: Ashequr Rahman 
"""
#=============================================================================================================================
## Import Libraries  ##
#=============================================================================================================================
import sys, os, gc
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import h5py
import time
import argparse
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, Dropout, BatchNormalization, concatenate, Add, Activation, LeakyReLU
from tensorflow.keras.activations import softmax
from tensorflow.keras.initializers import Constant

#import network model
from demist_model import build_dennet3D, ClearMemory
plt.switch_backend('Agg')
tf.compat.v1.enable_eager_execution()
#tf.config.run_functions_eagerly(True)

def log_plot_loss_metric(
                train_metric, val_metric,
                train_label, val_label,
                ylabel_name, fig_title,
                png_filename
              ): 

  fig, ax = plt.subplots()
  ax.plot(train_metric, label=train_label)
  ax.plot(val_metric, label=val_label)
  ax.set_yscale('log')
  ax.set_xlabel('epoch')
  ax.set_ylabel(ylabel_name)
  ax.set_title(fig_title)
  ax.legend()

  plt.savefig(png_filename, bbox_inches='tight') 
  plt.close(fig)

  return
def get_indices(train_pat_index, test_pat_index, num_data_dict, loc_arr, ext_arr, sev_arr):
  train_index = []
  test_index = []

  j = 0
  for hd in ['hl']:
    for ind_pat in range(num_data_dict[hd]['start_ind'], num_data_dict[hd]['end_ind'] + 1):
      for loc in loc_arr:
        for ext in ext_arr:
          if (ind_pat in train_pat_index):
            train_index.append(j)
          else:
            test_index.append(j)
          j = j + 1

  for hd in ['def']:
    for ind_pat in range(num_data_dict[hd]['start_ind'], num_data_dict[hd]['end_ind'] + 1):
      for loc in loc_arr:
        for ext in ext_arr:
          for sev in sev_arr:
            if (ind_pat in train_pat_index):
              train_index.append(j)
            else:
              test_index.append(j)
            j = j + 1

  return train_index, test_index



def my_read_bin(cur_inp_file, data_type, input_shape):
  A = np.fromfile(cur_inp_file, dtype = data_type)
  A[np.isnan(A)] = 0
  A = np.reshape(A, input_shape)
  A = np.transpose(A, [2, 1, 0, 3])
  return A

#@tf.function
def train_step(model, x_batch_train, y_batch_train, y_batch_loc_train, opt):
  with tf.GradientTape() as tape:
    model([x_batch_train, y_batch_train, y_batch_loc_train], training=True)
    loss_value = sum(model.losses)

  grads = tape.gradient(loss_value, model.trainable_weights)
  opt.apply_gradients(zip(grads, model.trainable_weights))
  return loss_value
   
#@tf.function
def val_step(model, x_batch_val, y_batch_val,y_batch_loc_val):
  # forward pass multipe times to get IW estimate
  model([x_batch_val, y_batch_val, y_batch_loc_val], training=False)
  loss_value = sum(model.losses)
  return loss_value

#=============================================================================================================================
## Define training protocols  ##
#=============================================================================================================================
def train(base_folder, weights_name, loss_fn_name, dose_level, num_iter, bt_size, num_epochs, learning_folder, 
                lambda_val_ind_chdiff):

  #======================================================================================================================================
  # 1. Data pre-processing
  #======================================================================================================================================
  lambda_val_arr_chdiff = [0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1]
  lambda_val_chdiff = lambda_val_arr_chdiff[lambda_val_ind_chdiff]

  data_folder_def = f'path/to/train/data/for/defect-present/cases'
  data_folder_hl = f'path/to/train/data/for/defect-absent/cases'
  data_folder_centroids =f'path/to/defect-centroid/data'
  result_folder = f'path/for/saving'

  # load protocols
  dose_level_max = 1
  num_reg = 1
  Nx_in, Ny_in, Nz_in = 48, 48, 48
  num_input_channels = 1
  num_output_channels = num_reg
  Nx_out, Ny_out, Nz_out = 48, 48, 48

  input_shape = (Nx_in, Ny_in, Nz_in, num_input_channels)
  loc_shape = (Nx_in, Ny_in, 1)
  input_shape_orig = (Nz_in, Ny_in, Nx_in, num_input_channels)
  output_shape = (Nx_out, Ny_out, Nz_out, num_output_channels)
  output_shape_orig = (Nz_out, Ny_out, Nx_out, num_output_channels)

  num_pat = 184
  num_data_dict = {
        'hl': {
                'start_ind': 0,
                'end_ind': num_pat-1,
              },
        'def': {
                'start_ind': 0,
                'end_ind': num_pat-1,
              }
      }
  loc_arr = ['a', 'i']
  sev_arr = [100, 175, 250]
  ext_arr = [30, 60]

  num_train_hl = num_data_dict['hl']['end_ind'] - num_data_dict['hl']['start_ind'] + 1
  num_train_def = num_data_dict['def']['end_ind'] - num_data_dict['def']['start_ind'] + 1
  num_train = num_train_hl*len(loc_arr)*len(ext_arr) + num_train_def * len(loc_arr)* len(ext_arr) * len(sev_arr)
  X_data = np.zeros((num_train,) + input_shape)
  Y_data = np.zeros((num_train,) + output_shape)
  Y_data_loc = np.zeros((num_train,) + loc_shape)

  pat_id_arr_fname = f'path/to/list/of/patient/id/.txt'
  pat_id_arr = np.loadtxt(pat_id_arr_fname, dtype = 'str', comments="#", delimiter=",", unpack=False)


  sc_data = 1e2
  j = 0
  for hd in ['hl']:
    for ind_pat in range(num_data_dict[hd]['start_ind'], num_data_dict[hd]['end_ind'] + 1):
      cur_inp_file = #f'{data_folder_hl}/{pat_id_arr[ind_pat]}/{hd}/recon_pat{pat_id_arr[ind_pat]}_d{dose_level}_it{num_iter}_c30o5.img'
      cur_label_file = #f'{data_folder_hl}/{pat_id_arr[ind_pat]}/{hd}/recon_pat{pat_id_arr[ind_pat]}_d{dose_level_max}_it{num_iter}_c30o5.img'
      cur_X = my_read_bin(cur_inp_file, 'float32', input_shape_orig)
      cur_Y = my_read_bin(cur_label_file, 'float32', input_shape_orig)
      cur_X = (cur_X - np.min(cur_X))/(np.max(cur_X) - np.min(cur_X))*sc_data
      cur_Y = (cur_Y - np.min(cur_Y))/(np.max(cur_Y) - np.min(cur_Y))*sc_data

      for loc in loc_arr:
        for ext in ext_arr:
          def_loc_fname = #f'{data_folder_centroids}/{pat_id_arr[ind_pat]}/def_centroid_d{loc}21{ext}_mod_apr3.bin'
          cur_loc = np.fromfile(def_loc_fname, dtype = 'float32').astype(int) - 1 + 1 #0 -based / but nn2D has 1 shift

          X_data[j,:, :, :, :] = cur_X
          Y_data[j,:, :, :, :] = cur_Y
          Y_data_loc[j, cur_loc[1] , cur_loc[0], 0] = 1.0
          j = j + 1

  for hd in ['def']:
    for ind_pat in range(num_data_dict[hd]['start_ind'], num_data_dict[hd]['end_ind'] + 1):
      for loc in loc_arr:
        for ext in ext_arr:
          def_loc_fname = #f'{data_folder_centroids}/{pat_id_arr[ind_pat]}/def_centroid_d{loc}21{ext}_mod_apr3.bin'
          cur_loc = np.fromfile(def_loc_fname, dtype = 'float32').astype(int) - 1 + 1
          for sev in sev_arr:
            def_name = f'd{loc}21{ext}s{sev}'
            cur_inp_file = #f'{data_folder_def}/{pat_id_arr[ind_pat]}/{def_name}/recon_pat{pat_id_arr[ind_pat]}_d{dose_level}_it{num_iter}_c30o5.img'
            cur_label_file = #f'{data_folder_def}/{pat_id_arr[ind_pat]}/{def_name}/recon_pat{pat_id_arr[ind_pat]}_d{dose_level_max}_it{num_iter}_c30o5.img'
            cur_X = my_read_bin(cur_inp_file, 'float32', input_shape_orig)
            cur_Y = my_read_bin(cur_label_file, 'float32', input_shape_orig)
            cur_X = (cur_X - np.min(cur_X))/(np.max(cur_X) - np.min(cur_X))*sc_data
            cur_Y = (cur_Y - np.min(cur_Y))/(np.max(cur_Y) - np.min(cur_Y))*sc_data

            X_data[j,:, :, :, :] = cur_X
            Y_data[j,:, :, :, :] = cur_Y
            Y_data_loc[j, cur_loc[1] , cur_loc[0], 0] = 1.0
            j = j + 1

  X_data = X_data[:j, :, :, :, :]
  Y_data = Y_data[:j, :, :, :, :]
  Y_data_loc = Y_data_loc[:j, :]

  print("--------------------------------------------------------")
  print("Training Input Shape => " + str(X_data.shape))
  print("Training Label Shape => " + str(Y_data.shape))
  print("Training Location input Shape => " + str(Y_data_loc.shape))
  print('Number of Training Examples: %d' % X_data.shape[0])
  print("--------------------------------------------------------")

  #======================================================================================================================================
  # 2. Import channels
  #======================================================================================================================================
  channel_fname = f'/path/to/anthropomorphic/channels/U_64.npy'
  U = np.load(channel_fname)
  ch_dim, num_ch = 64, 4
  U = np.reshape(U, [ch_dim, ch_dim, 1, num_ch])
  U = tf.convert_to_tensor(U, np.float32)
  num_fold = 4
  rand_state = 1

  #===================================
  # clear
  #===================================
  X_train = None
  X_test = None
  Y_train = None
  Y_test = None
  Y_train_loc = None
  Y_test_loc = None
  model = None
  checkpoint = None
  train_history = None
  csv_logger = None
  K.clear_session();
  gc.collect();

  #===================================
  # set up fold data
  #===================================
  train_dataset = tf.data.Dataset.from_tensor_slices((X_data, Y_data, Y_data_loc))
  train_dataset = train_dataset.shuffle(buffer_size=len(X_data)).batch(bt_size)
  val_dataset = tf.data.Dataset.from_tensor_slices((X_data, Y_data, Y_data_loc)).batch(bt_size)

  #===================================
  # custom loop
  #===================================
  denoise_obs_net = build_dennet3D(input_shape, loc_shape, num_reg, lambda_val_chdiff, U)
  com_flag = f'_d{dose_level}_it{num_iter}_b{bt_size}_lmbdchdiff{lambda_val_ind_chdiff}'
  weights_base_name = f'{result_folder}/weights/{weights_name}' + com_flag
  opt = tf.keras.optimizers.Adam(learning_rate=1e-3) #change

  logs = {}
  logs_it = {}
  val_flag = 1

  for epoch in range(num_epochs):
    start_time = time.time()
    print(f'Epoch:{epoch + 1}/{num_epochs}')
    #=====================================================================
    # step-1: Training steps
    #=====================================================================
    denoise_obs_net.reset_metrics()
    loss_value = 0
    for step, (x_train_batch, y_train_batch, y_train_loc_batch) in enumerate(train_dataset):
      cur_loss = train_step(denoise_obs_net, x_train_batch, y_train_batch, y_train_loc_batch, opt)
      loss_value += cur_loss
      for m in denoise_obs_net.metrics:
        if 'train_' + m.name in logs_it:
          logs_it['train_' + m.name].append(m.result())
        else:
          logs_it['train_' + m.name] = [m.result()]

    #=====================================================================
    # step-2: log all losses and metrics on each epoch end of training set
    #=====================================================================
    for m in denoise_obs_net.metrics:
      if 'train_' + m.name in logs:
        logs['train_' + m.name].append(m.result())
      else:
        logs['train_' + m.name] = [m.result()]

    cur_train_loss = loss_value / (step + 1)
    if 'train_loss' in logs:
      logs['train_loss'].append(cur_train_loss)
    else:
      logs['train_loss'] = [cur_train_loss]

    #=====================================================================
    # step-3: log metric in terminal
    #=====================================================================
    print(
            f'Train: loss={cur_train_loss:.3f} || ' \
            f"mse = {logs['train_mse'][-1]:.3f} || " \
            f"chv_mse = {logs['train_chv_mse'][-1]:.3f} || " \
        )

    print('=' * 60)

    #=====================================================================
    # step-6: save model, relevant metrics and loss function
    #=====================================================================
    save_freq = 20 #change
    if (epoch + 1) % num_epochs == 0:
      weights_filename = f'{weights_base_name}_ep{epoch+1:03d}.hdf5'
      denoise_obs_net.save_weights(
                          weights_filename,
                          save_format='h5',
                        )
  #-----------------------------------------------------------
  # Step-3: Save model and loss curve
  #----------------------------------------------------------
  # train and validation losses
  train_loss = logs['train_loss']

  # train and validation metrics
  # MSE
  train_mse = logs['train_mse']

  # chv_mse
  train_chv_mse = logs['train_chv_mse']


  mdict = {
            "train_loss": train_loss,
            "train_mse": train_mse,
            "train_chv_mse": train_chv_mse,
          }
  loss_curve_name = loss_fn_name + com_flag

  sio.savemat(f'{result_folder}/losses/{loss_curve_name}.mat', mdict)
  np.save(f'{result_folder}/losses/{loss_curve_name}.npy', mdict)

  sio.savemat(f'{result_folder}/losses/{loss_curve_name}_it.mat', logs_it)
  #-----------------------------------------------------------
  # EOF!!!
  #----------------------------------------------------------

  return

#=============================================================================================================================
## Main  ##
#=============================================================================================================================
if __name__ == '__main__':
  # start a parser
  parser = argparse.ArgumentParser()
  parser.add_argument('--weights_name', type=str, help="path to save model weights as hdf5-file")
  parser.add_argument('--loss_fn_name', type=str, help="path to save loss curves as m-file")
  parser.add_argument('--base_folder', type=str, help="base folder for data and saving denoised images")
  parser.add_argument('--dose_level', type=int, help="dose_level")
  parser.add_argument('--num_iter', type=int, help="number of iterations in reconstructions")
  parser.add_argument('--batch_size', type=int, help="batch_size in learning scheme")
  parser.add_argument('--epochs', type=int, help="epochs in learning scheme")
  parser.add_argument('--learning_folder', type=str, help="saving folder of learning")
  parser.add_argument('--lambda_val_ind_chdiff', type=int, help="lambda value index of channel vector difference loss")

  # parse the input arguments
  args = parser.parse_args()

  # Launch training routine
  train(args.base_folder, args.weights_name, args.loss_fn_name, args.dose_level, args.num_iter, 
        args.batch_size, args.epochs, args.learning_folder,
        args.lambda_val_ind_chdiff)


