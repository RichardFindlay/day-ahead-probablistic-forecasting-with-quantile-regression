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



model_type ="wind"

if model_type == 'wind':
	dataset_name = 'train_set_V100_wind.hdf5'
elif model_type == 'demand':
	dataset_name = 'dataset_V2_withtimefeatures_Demand.hdf5'
elif model_type == 'solar':
	dataset_name = 'dataset_solar_v30.hdf5'
elif model_type == 'price':
	dataset_name = 'dataset_V1_DAprice.hdf5'


# collect param sizes
f = h5py.File(f"./Data/{model_type}/Processed_Data/{dataset_name}", "r")
features = np.empty_like(f['train_set']['X1_train'][0])
times_in = np.empty_like(f['train_set']['X2_train'][0])
times_out = np.empty_like(f['train_set']['X3_train'][0])
labels = np.empty_like(f['train_set']['y_train'][0])
x_len = f['train_set']['X1_train'].shape[0]
y_len = f['train_set']['y_train'].shape[0]
print('size parameters loaded')

height, width, channels = features.shape[0], features.shape[1], features.shape[2]
times_in_dim = times_in.shape[-1]
times_out_dim = times_out.shape[-1]


quantiles = [0.1, 0.5, 0.9]

Tx = 336
Ty = 48
n_s = 32

input_seq_size = Tx
output_seq_size =Ty

# create swish activation object
from keras.backend import sigmoid

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

# Getting the Custom object and updating them
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
  
# Below in place of swish you can take any custom key for the name 
get_custom_objects().update({'swish': Activation(swish)})

print('reading models')
model = load_model(f'./Models/{model_type}_models/q_all/{model_type}Generation_forecast_MainModel_Q_all.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention, 'Activation': Activation(swish)})

# read encoder models
temporal_enc = load_model(f'./Models/{model_type}_models/q_all/{model_type}Generation_encoderModelTemporal_Q_all.h5') 
spatial_enc = load_model(f'./Models/{model_type}_models/q_all/{model_type}Generation_encoderModelSpatial_Q_all.h5') 


# load and process data
f = h5py.File(f"./Data/{model_type}/Processed_Data/{dataset_name}", "r")

set_type = 'test'

X_train1 = f[f'{set_type}_set'][f'X1_{set_type}'][0:3000]
X_train2 = f[f'{set_type}_set'][f'X2_{set_type}'][0:3000]
X_train3 = f[f'{set_type}_set'][f'X3_{set_type}'][0:3000]
X_train4 = f[f'{set_type}_set'][f'X1_{set_type}'][0:3000]
y_train = f[f'{set_type}_set'][f'y_{set_type}'][0:3000]


# X_train1 = f['test_set']['X1_test'][:500]
# X_train2 = f['test_set']['X2_test'][:500]
# X_train3 = f['test_set']['X3_test'][:500]
# y_train = f['test_set']['y_test'][:500]

print(X_train1.shape)

# decoder_input = f['train_set']['y_train'][output_indexes]

# f.close()  

input_start, output_start = 0, input_seq_size

seqX1, seqX2, seqX3, seqX4, seqY = [], [], [], [], []

# a = np.array(X_train1)

while (output_start + output_seq_size) <= len(y_train):
	# offset handled during pre-processing
	input_end = input_start + input_seq_size
	output_end = output_start + output_seq_size

	# inputs
	seqX1.append(X_train1[input_start:input_end])
	seqX2.append(X_train2[input_start:input_end])

	# outputs
	seqX3.append(X_train3[output_start:output_end])
	a = X_train4[output_start:output_end][:,:,:,1:]
	a = np.average(a, axis=(1,2))
	seqX4.append(a)
	seqY.append(y_train[output_start:output_end])

	input_start += output_seq_size
	output_start += output_seq_size


x1, x2, x3, x4, y = np.array(seqX1), np.array(seqX2), np.array(seqX3), np.array(seqX4), np.array(seqY)
f.close() 

print(x1.shape)
print(y.shape)

s0 = np.zeros((1, n_s))
c0 = np.zeros((1, n_s))






# function for decoder model
def inference_dec_model(quantile):

	# Encoder outputs for setup
	ccn_enc_output_test = Input(shape=(320, 128))
	lstm_enc_output_test = Input(shape=(Tx, n_s*4)) #+ times_in_dim

	# Decoder Input
	times_in = Input(shape=(1, times_in_dim))
	times_out = Input(shape=(1, times_out_dim))
	out_nwp = Input(shape=(1, channels-1))
	s_state0 = Input(shape=(32,))
	c_state0 = Input(shape=(32,))
	decoder_input = Input(shape=(1, 11))

	enc_out = concatenate([out_nwp, times_out], axis=-1)

	# context and previous output
	attn_weights_temp_test, context_temp_test = model.get_layer(f'temporal_attention_q_{quantile}')(lstm_enc_output_test, enc_out, s_state0, c_state0)
	attn_weights_spat_test, context_spat_test = model.get_layer(f'spatial_attention_q_{quantile}')(ccn_enc_output_test, enc_out, s_state0, c_state0)

	# context & previous output combine
	context = concatenate([context_temp_test, context_spat_test], axis=-1) 

	# decoder_input = concatenate([out_nwp, times_out])

	# Decoder inference
	dec_output, s_state, c_state = model.get_layer(f'decoder_q_{quantile}')(decoder_input, initial_state=[s_state0, c_state0])

	# combine context and prediction
	prediction = concatenate([context, K.expand_dims(dec_output,axis=1)])

	# final dense layer
	pred_test = model.get_layer(f'dense1_q_{quantile}')(prediction)
	pred_test = model.get_layer(f'dense3_q_{quantile}')(pred_test)

	# Inference Model
	deoceder_test_model = Model(inputs=[times_in, times_out, out_nwp, decoder_input, ccn_enc_output_test, lstm_enc_output_test, s_state0, c_state0], outputs=[pred_test, s_state, c_state, attn_weights_temp_test, attn_weights_spat_test])  
	     
	return deoceder_test_model

# dictionary to store decoder models
decoder_models = {}

# instantiate model for each quantile
for q in quantiles:
	decoder_models[f'{q}'] = inference_dec_model(q)

# store predictions
predictions = {}

# np.zeros((x1.shape[0], Ty, 1))



# loop through each sample, passing individually to model
for q in quantiles:
	print(q)

	# set hidden states to zero
	# s_state, c_state = s0, c0

	total_pred = np.empty((x1.shape[0], Ty, 1))

	decoder = decoder_models[f'{q}']

	for idx in range(x1.shape[0]): # loop through each sample, to keep track of hidden states

		outputs = []

		s_state, c_state = s0, c0

		# create final inference model
		lstm_enc_output, enc_s_state, enc_c_state = temporal_enc([x1[idx:idx+1], x2[idx:idx+1]])
		ccn_enc_output = spatial_enc(x1[idx:idx+1])

		intial_in = np.average(x1[idx:idx+1], axis=(2,3))

		for ts in range(Ty):

			# declare decoder input 
			if ts > 0:
				decoder_input = concatenate([x4[idx:idx+1,ts-1:ts,:], x3[idx:idx+1,ts-1:ts,:]], axis=-1)
			else:
				decoder_input = concatenate([intial_in[:,-1:,1:], x2[idx:idx+1,-1:,:]], axis=-1)  


			pred, s_state, c_state, attn_weights_temp_test, attn_weights_spat_test = decoder([x2[idx:idx+1,ts:ts+1,:], x3[idx:idx+1,ts:ts+1,:], x4[idx:idx+1,ts:ts+1,:], decoder_input, ccn_enc_output, lstm_enc_output, s_state, c_state])

			outputs.append(pred)

		combined_outputs = np.concatenate(outputs, axis=1)

		total_pred[idx, : , :] = combined_outputs

	predictions[f'{q}'] = total_pred


plot_ref = 0

# plot predictions
for idx, (key, values) in enumerate(predictions.items()):
	plt.plot(values[plot_ref:plot_ref+7,:].flatten(), label=f"prediction_{key}")

plt.plot(y[plot_ref:plot_ref+7,:,0].flatten(), label="actual")
plt.legend()
plt.show()













