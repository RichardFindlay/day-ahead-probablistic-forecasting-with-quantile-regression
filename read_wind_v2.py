import pandas as pd
import numpy as np
import netCDF4
from netCDF4 import Dataset
import os 
import glob
import sys
# import tensorflow as tf
from collections import OrderedDict
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model, model_from_json
from keras import Model
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv1D, Conv2D, Softmax, Bidirectional, Dense, TimeDistributed, LSTM 
from tensorflow.keras.layers import Input, Activation, AveragePooling2D, Lambda, concatenate, Flatten, BatchNormalization, RepeatVector, Permute, Lambda, Dropout
from tensorflow.keras.layers import Reshape
from keras.callbacks import ModelCheckpoint
from pickle import load
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import seaborn as sns
from pickle import dump, load

import geopandas
import contextily as ctx

from attentionlayer import attention
import sys
import h5py








def inference_model(quantile):

	# LSTM Encoder
	# enc_model_temp_test = Model(inputs = [x_input, times_in], outputs=[lstm_enc_output])
	# CNN Encoder
	# enc_model_spat_test = Model(x_input, ccn_enc_output) 

	# Encoder outputs for setup
	ccn_enc_output_test = Input(shape=(320, 128))
	lstm_enc_output_test = Input(shape=(Tx, n_s)) #+ times_in_dim

	# Decoder Input
	dec_input_test = Input(shape=(1, None))
	dec_input_test_int = Input(shape=(1, 1)) #+ times_in_dim
	times_out_test = Input(shape=(1, times_out_dim))

	# context and previous output
	attn_weights_temp_test, context_temp_test = model.get_layer('temporal_attention')(lstm_enc_output_test, s_state0, c_state0)
	attn_weights_spat_test, context_spat_test = model.get_layer('spatial_attention')(ccn_enc_output_test, s_state0, c_state0)

	# context & previous output combine
	context_test = concatenate([context_spat_test, context_temp_test], axis=-1) 
	dec_input_concat_test_int = concatenate([context_test, dec_input_test_int], axis=-1)

	# combine with decoder inputs
	dec_input_concat_test = concatenate([context_test, times_out_test], axis=-1)
	dec_input_concat_test_int = concatenate([dec_input_concat_test_int, times_out_test], axis=-1)

	dec_input_concat_test = concatenate([dec_input_concat_test, dec_input_test], axis=-1)

	# Decoder inference
	# if idx == 1:
	# 	dec_output, s_state, c_state = model.get_layer('lstm_1')(dec_input_concat_test_int, initial_state=[s_state0, c_state0])
	# else:   
	# 	dec_output, s_state, c_state = model.get_layer(f'lstm_{idx}')(dec_input_concat_test, initial_state=[s_state0, c_state0])

	dec_output, s_state, c_state = model.get_layer('lstm_1')(dec_input_concat_test, initial_state=[s_state0, c_state0])

	# pred_test = model.get_layer(f'1conv_{idx-1}')(dec_output)
	# pred_test = model.get_layer(f'2conv_{idx-1}')(pred_test)
	# pred_test = model.get_layer(f'3conv_{idx-1}')(pred_test)


	pred_test = model.get_layer('conv1d')(dec_output)
	pred_test = model.get_layer('conv1d_1')(pred_test)
	pred_test = model.get_layer('conv1d_2')(pred_test)

	# Inference Model
	deoceder_test_model = Model(inputs=[dec_input_test, times_out_test, lstm_enc_output_test, ccn_enc_output_test, s_state0, c_state0], outputs=[pred_test, s_state, c_state, attn_weights_temp_test, attn_weights_spat_test])  
	     
	return deoceder_test_model







