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






type ="demand"

if type == 'wind':
	dataset_name = 'train_set_V6_withtimefeatures_120hrinput_float32.hdf5'
elif type == 'demand':
	dataset_name = 'dataset_V1_withtimefeatures_Demand.hdf5'
elif type == 'solar':
	dataset_name = 'train_set_V21_withtimefeatures_120hrinput.hdf5'

# dataset_name = 'train_set_V6_withtimefeatures_120hrinput_float32.hdf5'
f = h5py.File(f"./Data/{type}/Processed_Data/{dataset_name}", "r")
features = np.empty_like(f['train_set']['X1_train'][0])
times_in = np.empty_like(f['train_set']['X2_train'][0])
times_out = np.empty_like(f['train_set']['X3_train'][0])
labels = np.empty_like(f['train_set']['y_train'][0])
x_len = f['train_set']['X1_train'].shape[0]
y_len = f['train_set']['y_train'].shape[0]
print('size parameters loaded')
f.close()  

input_seq_size = 672
output_seq_size = 48
n_s = 128

# time_set_load = open(f"./Data/{type}/Processed_Data/time_refs_V6_withtimefeatures_120hrinput.pkl", "rb") 
# time_set = load(time_set_load)
# time_set_load.close()

# idx =0
# print('time check')
# print(len(time_set['input_times_test']))
# print(len(time_set['output_times_test']))
# print(time_set['input_times_test'])
# print('*************************************************')
# print(time_set['output_times_test'][265:])

# make custom activation - swish
from keras.backend import sigmoid

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

# Getting the Custom object and updating them
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
  
# Below in place of swish you can take any custom key for the name 
get_custom_objects().update({'swish': Activation(swish)})



# split test data into sequences
f = h5py.File(f"./Data/{type}/Processed_Data/{dataset_name}", "r")


X_train1 = f['test_set']['X1_test'][500:4500]
X_train2 = f['test_set']['X2_test'][500:4500]
X_train3 = f['test_set']['X3_test'][500:4500]
X_train4 = f['test_set']['X1_test'][500:4500]
y_train = f['test_set']['y_test'][500:4500]


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

s0 = np.zeros((x1.shape[0], n_s))
c0 = np.zeros((x1.shape[0], n_s))


model = load_model(f'./Models/{type}_models/q_0.5/{type}Generation_forecast_MainModel_Q_0.5.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention, 'Activation': Activation(swish)})
# model1 = load_model(f'./Models/{type}_models/q_0.01/{type}Generation_forecast_MainModel_Q_0.01.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention, 'Activation': Activation(swish)})
# model2 = load_model(f'./Models/{type}_models/q_0.99/{type}Generation_forecast_MainModel_Q_0.99.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention, 'Activation': Activation(swish)})
# print(model.summary())
# print(model.layers[-2].get_config())
# exit()
# # x1 = np.average(x1, axis=(2,3))
# # x1 = x1[:, :, 0:1]

# print(x1.shape)
# print(c0.shape)

# predictions = model.predict([x1, x2, x3, x4, s0, c0])
# predictions1 = model1.predict([x1, x2, x3, x4, s0, c0])
# predictions2 = model2.predict([x1, x2, x3, x4, s0, c0])

# predictions = predictions[0]
# predictions1 = predictions1[0]
# predictions2 = predictions2[0]

# idx = 20
# plt.plot(predictions[idx:idx+7,:].flatten(), label="prediction_0.5")
# plt.plot(predictions1[idx:idx+7,:].flatten(), label="prediction_0.1")
# plt.plot(predictions2[idx:idx+7,:].flatten(), label="prediction_0.9")
# plt.plot(y[idx:idx+7,:,0].flatten(), label="actual")
# plt.legend()
# plt.show()



############################################################################################################################################################################### 

Tx = input_seq_size
Ty = output_seq_size
height, width, channels = features.shape[0], features.shape[1], features.shape[2]
times_in_dim = times_in.shape[-1]
times_out_dim = times_out.shape[-1]
n_s = 128


enoder_temporal_model = load_model(f'./Models/{type}_models/q_0.5/{type}Generation_encoderModel_temporal_Q_0.5.h5')
enoder_spatial_model = load_model(f'./Models/{type}_models/q_0.5/{type}Generation_encoderModel_spatial_Q_0.5.h5')


def inference_model():

	# Encoder outputs for setup
	ccn_enc_output_test = Input(shape=(1584))
	lstm_enc_output_test = Input(shape=(n_s)) #+ times_in_dim

	# Decoder Inputs
	nwp_out = Input(shape=(Ty, channels-1))
	times_out = Input(shape=(Ty, times_out_dim))

	# hidden state inputs
	s_state_in = Input(shape=(n_s))
	c_state_in = Input(shape=(n_s))

	# encoder combine
	encoder_out_combine = concatenate([ccn_enc_output_test, lstm_enc_output_test], axis=-1) 
	encoder_out_combine = RepeatVector(Ty)(encoder_out_combine)

	# combine with decoder inputs
	dec_input_concat = concatenate([encoder_out_combine, nwp_out], axis=-1)
	dec_input_concat = concatenate([dec_input_concat, times_out], axis=-1)

	dec_output, s_state, c_state = model.get_layer('lstm_1')(dec_input_concat, initial_state=[s_state_in, c_state_in])

	pred_test = model.get_layer('dense')(dec_output)
	pred_test = model.get_layer('dense_2')(pred_test)

	# Inference Model
	deoceder_test_model = Model(inputs=[ccn_enc_output_test, lstm_enc_output_test, times_out, nwp_out, s_state_in, c_state_in], outputs=[pred_test, s_state, c_state])  
	     
	return deoceder_test_model

# declare decoder model
decoder_model = inference_model()


# def predict(input1, input2, nwp_out, times_out, s_state, c_state):
# 	# spatial encoder
# 	ccn_enc_output = enoder_spatial_model.predict(input1)

# 	# lstm encoder
# 	encoder_out, s_state, c_state = enoder_temporal_model.predict([input1, input2])

# 	# call decoder
# 	prediction, s_state, c_state = decoder_model.predict([ccn_enc_output_test, lstm_enc_output_test, times_out, nwp_out,  s_state, c_state])

# 	return prediction




next_input = x1[0:1,:,:,:,0:1] 
broadcaster = np.ones((1, output_seq_size, next_input.shape[2], next_input.shape[3], 1), dtype=np.float32)

for sample in range(20, 30):
	print(sample)

	# x1[sample:sample+1,:,:,:,0:1] = next_input 

	# spatial encoder
	ccn_enc_output = enoder_spatial_model.predict(x1[sample:sample+1,:,:,:,:])

	# lstm encoder
	encoder_out, enc_s_state, enc_c_state = enoder_temporal_model.predict([x1[sample:sample+1,:,:,:,:], x2[sample:sample+1,:,:]])

	if sample == 20:
		s_state, c_state= enc_s_state, enc_c_state


	# call decoder
	# [x1[sample:sample+1,:,:,:,:], x2[sample:sample+1,:,:], x3[sample:sample+1,:,:], x4[sample:sample+1,:,:], s0[sample:sample+1,:], c0[sample:sample+1,:]]
	prediction, s_state, c_state = decoder_model.predict([ccn_enc_output, encoder_out, x3[sample:sample+1,:,:], x4[sample:sample+1,:,:],  s_state, c_state])
	

	if sample == 20:
		predictions = prediction
	else:
		predictions = np.concatenate([predictions, prediction], axis=0)


	# prediction_transform =  broadcaster * np.expand_dims(np.expand_dims(prediction, axis=-1), axis=-1)
	# next_input = np.concatenate([next_input, prediction_transform], axis=1)[0:1, -input_seq_size:, :, :, 0:1]


idx = 0
plt.plot(predictions[idx:idx+7,:].flatten(), label="prediction_0.5")
# plt.plot(predictions1[idx:idx+7,:].flatten(), label="prediction_0.1")
# plt.plot(predictions2[idx:idx+7,:].flatten(), label="prediction_0.9")
plt.plot(y[20:20+7,:,0].flatten(), label="actual")
plt.legend()
plt.show()











