""" @Author: Ashequr Rahman
    @Date: 2022-05-22  
    @Last Modified by: Ashequr Rahman 
"""
#=============================================================================================================================
## Import Libraries  ##
#=============================================================================================================================
import sys, os
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
from demist_model import build_dennet3D_pred

def my_write_bin(cur_out_file, data_type, data):
  data = np.transpose(data, [2, 1, 0])
  data.astype(data_type).tofile(cur_out_file)
  return
def my_read_bin(cur_inp_file, data_type, input_shape):
  A = np.fromfile(cur_inp_file, dtype = data_type)
  A[np.isnan(A)] = 0
  A = np.reshape(A, input_shape)
  A = np.transpose(A, [2, 1, 0, 3])
  return A


#=============================================================================================================================
## Define training protocols  ##
#=============================================================================================================================
def test(base_folder, weights_name, loss_fn_name, dose_level, num_iter, bt_size, num_epochs, learning_folder,
                lambda_val_ind_chdiff, file_name):
  #-----------------------------------------------------------
  # Step-1: Data pre-processing
  #----------------------------------------------------------
  lambda_val_arr_chdiff = [0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1]
  lambda_val_chdiff = lambda_val_arr_chdiff[lambda_val_ind_chdiff]

  result_folder = f'{base_folder}/learning/{learning_folder}'

  # load protocols
  dose_level_max = 1
  num_z_slice = 64
  num_reg = 1
  Nx_in, Ny_in, Nz_in = 48, 48, 48
  num_input_channels = 1
  num_output_channels = num_reg
  Nx_out, Ny_out, Nz_out = 48, 48, 48

  input_shape = (Nx_in, Ny_in, Nz_in, num_input_channels)
  input_shape_orig = (Nz_in, Ny_in, Nx_in, num_input_channels)
  output_shape = (Nx_out, Ny_out, Nz_out, num_output_channels)

  num_test = 1
  X_test = np.zeros((num_test,) + input_shape)

  pat_id_arr_fname = f'{base_folder}/test_pat_list.txt'
  pat_id_arr = np.loadtxt(pat_id_arr_fname, dtype = 'str', comments="#", delimiter=",", unpack=False)

  sc_data = 1e2
  cur_inp_file = file_name #
  cur_X = my_read_bin(cur_inp_file, 'float32', input_shape_orig)
  cur_X = (cur_X - np.min(cur_X))/(np.max(cur_X) - np.min(cur_X))*sc_data
  X_test[0,:, :, :, :] = cur_X[:, :, :, :]

  #-----------------------------------------------------------
  # Step-2: load model and predict
  #----------------------------------------------------------
  pred_model = build_dennet3D_pred(input_shape, num_reg)
  com_flag = f'_d{dose_level}_it{num_iter}_b{bt_size}_lmbdchdiff{lambda_val_ind_chdiff}'
  weights_base_name = f'{result_folder}/weights/{weights_name}{com_flag}_ep{num_epochs:03d}.hdf5'
  pred_model.load_weights(weights_base_name, by_name = True)
  Y_pred = pred_model.predict(X_test, batch_size=1)

  return

#=============================================================================================================================
## Main  ##
#=============================================================================================================================
if __name__ == '__main__':
  # start a parser
  parser = argparse.ArgumentParser()
  parser.add_argument('--weights_name', type=str, help="path to saved model weights as hdf5-file")
  parser.add_argument('--loss_fn_name', type=str, help="path to saved loss curves as m-file")
  parser.add_argument('--base_folder', type=str, help="base folder for data and saving denoised images")
  parser.add_argument('--dose_level', type=int, help="dose_level")
  parser.add_argument('--num_iter', type=int, help="number of iterations in reconstructions")
  parser.add_argument('--batch_size', type=int, help="batch_size in learning scheme")
  parser.add_argument('--epochs', type=int, help="epochs in learning scheme")
  parser.add_argument('--learning_folder', type=str, help="saving folder of learning")
  parser.add_argument('--lambda_val_ind_chdiff', type=int, help="lambda value index of channel vector difference loss")
  parser.add_argument('--file_name', type=str, help="file name of data to be denoised")




  # parse the input arguments
  args = parser.parse_args()
  # Launch training routine
  test(args.base_folder, args.weights_name, args.loss_fn_name, args.dose_level, args.num_iter, args.batch_size, args.epochs, args.learning_folder,
                args.lambda_val_ind_chdiff, args.file_name)
 

