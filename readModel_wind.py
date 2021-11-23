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

time_set_load = open(f"./Data/{type}/Processed_Data/time_refs_V1_withtimefeatures_Demand.pkl", "rb") # demand
# time_set_load = open(f"./Data/{type}/Processed_Data/time_refs_V6_withtimefeatures_120hrinput.pkl", "rb") # wind

time_set = load(time_set_load)
time_set_load.close()

# idx =0
# print('time check')
# print(len(time_set['input_times_test']))
# print(len(time_set['output_times_test']))
print(time_set['input_times_test'][0])
# print(time_set['input_times_test'][0:10])
print('*************************************************')
# print(time_set['output_times_test'][0:10])
print(time_set['output_times_test'][0])
exit()

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

set_type = 'train'

X_train1 = f[f'{set_type}_set'][f'X1_{set_type}'][0:2000]
X_train2 = f[f'{set_type}_set'][f'X2_{set_type}'][0:2000]
X_train3 = f[f'{set_type}_set'][f'X3_{set_type}'][0:2000]
X_train4 = f[f'{set_type}_set'][f'X1_{set_type}'][0:2000]
y_train = f[f'{set_type}_set'][f'y_{set_type}'][0:2000]


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
model1 = load_model(f'./Models/{type}_models/q_0.01/{type}Generation_forecast_MainModel_Q_0.01.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention, 'Activation': Activation(swish)})
model2 = load_model(f'./Models/{type}_models/q_0.99/{type}Generation_forecast_MainModel_Q_0.99.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention, 'Activation': Activation(swish)})
print(model2.summary())
print(model2.layers[-1].get_config())
# exit()
# # x1 = np.average(x1, axis=(2,3))
# # x1 = x1[:, :, 0:1]

# print(x1.shape)
# print(c0.shape)

predictions = model.predict([x1, x2, x3, x4, s0, c0])
predictions1 = model1.predict([x1, x2, x3, x4, s0, c0])
predictions2 = model2.predict([x1, x2, x3, x4, s0, c0])

# predictions = predictions[0]
# predictions1 = predictions1[0]
# predictions2 = predictions2[0]

idx = 0
plt.plot(predictions[idx:idx+7,:].flatten(), label="prediction_0.5")
plt.plot(predictions1[idx:idx+7,:].flatten(), label="prediction_0.1")
plt.plot(predictions2[idx:idx+7,:].flatten(), label="prediction_0.9")
plt.plot(y[idx:idx+7,:,0].flatten(), label="actual")
plt.legend()
plt.show()









# print(x1.shape)
# exit()


# output_len = 336

next_input = x1[0:1,:,:,:,0:1] 
broadcaster = np.ones((1, output_seq_size, next_input.shape[2], next_input.shape[3], 1), dtype=np.float32)

for sample in range(x1.shape[0]):

	x1[sample:sample+1,:,:,:,0:1] = next_input 

	prediction = model.predict([x1[sample:sample+1,:,:,:,:], x2[sample:sample+1,:,:], x3[sample:sample+1,:,:], x4[sample:sample+1,:,:], s0[sample:sample+1,:], c0[sample:sample+1,:]])
	prediction1 = model1.predict([x1[sample:sample+1,:,:,:,:], x2[sample:sample+1,:,:], x3[sample:sample+1,:,:], x4[sample:sample+1,:,:], s0[sample:sample+1,:], c0[sample:sample+1,:]])
	prediction2 = model2.predict([x1[sample:sample+1,:,:,:,:], x2[sample:sample+1,:,:], x3[sample:sample+1,:,:], x4[sample:sample+1,:,:], s0[sample:sample+1,:], c0[sample:sample+1,:]])

	if sample == 0:
		predictions = prediction
		predictions1 = prediction1
		predictions2 = prediction2
	else:
		predictions = np.concatenate([predictions, prediction], axis=0)
		predictions1 = np.concatenate([predictions1, prediction1], axis=0)
		predictions2 = np.concatenate([predictions2, prediction2], axis=0)

	# print(next_input.shape)
	# print(prediction.shape)

	prediction_transform =  broadcaster * np.expand_dims(np.expand_dims(prediction, axis=-1), axis=-1)
	# print(prediction_transform.shape)
	# print(next_input.shape)
	next_input = np.concatenate([next_input, prediction_transform], axis=1)[0:1, -input_seq_size:, :, :, 0:1]


	# print(next_input.shape)
	# exit()




# print(out.shape)
# exit()
# index = 0
# predictions = []
# for t in range(output_len):
# 	print(t)
# 	if t == 0:
# 		predictions = model.predict([x1[index:index+1,:,:], x2[index:index+1,:,:]])
# 	else:
# 		print(x1[index+t:index+1+t,:,:].shape)
# 		pred = model.predict([x1[index+t:index+1+t,:,:], x2[index+t:index+1+t,:,:]])

# 		a = np.concatenate([predictions, pred[:,-2:-1]], axis=-1)



# print(x1.shape)

# print(y.shape)
# exit()

# predictions = out

# idx = 22
plt.plot(predictions[idx:idx+7,:].flatten(), label="prediction_0.5")
plt.plot(predictions1[idx:idx+7,:].flatten(), label="prediction_0.1")
plt.plot(predictions2[idx:idx+7,:].flatten(), label="prediction_0.9")
plt.plot(y[idx:idx+7,:,0].flatten(), label="actual")
plt.legend()
plt.show()


def correlation_analysis(X, Y):

	rs = np.empty((X.shape[0], 1))
	#caclulate 'R^2' for each feature - average over all days
	for l in range(X.shape[0]):
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X[l,:,0], Y[l,:,0])
		rs[l, 0] =r_value**2
		

	print('mean' + '\n R**2: %s' %rs.mean())
	print('max' + '\n R**2: %s' %rs.max())
	print('min' + '\n R**2: %s' %rs.min())

	#get best
	best_fit = np.argmax(rs, axis=0)
	worst_fit = np.argmin(rs, axis=0)
	print(best_fit)
	print(worst_fit)
	# print(X[best_fit,:,0])

	return 

print(predictions.shape)
print(y.shape)

correlation_analysis(predictions, y)



exit()








params = {'batch_size': 32,
		'shuffle': False } 





class DataGenerator(tensorflow.keras.utils.Sequence):

	def __init__(self, dataset_name, x_length, y_length, batch_size, shuffle):
		self.dataset_name = dataset_name
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.xlen = x_length
		self.ylen = y_length 
		self.index_ref = 0         
		self.on_epoch_end()

	def __len__(self):
		# 'number of batches per Epoch'      
		# print(int(np.floor((self.xlen - (input_seq_size-1)) / self.batch_size)))
		# return int(np.floor((self.xlen - (input_seq_size-1)) / self.batch_size))
		return 5

	def __getitem__(self, index):

		# print(index)        

		input_indexes = self.input_indexes[(index*self.batch_size) : (index*self.batch_size)+ (self.batch_size + (input_seq_size-1))]
		output_indexes = self.output_indexes[(index*self.batch_size) : (index*self.batch_size) + (self.batch_size + (output_seq_size-1))]

		# self.index_ref_in += self.batch_size + (input_seq_size-1)
		# self.index_ref_out += self.batch_size + (output_seq_size-1)

		# Generate data
		(X_train1, X_train2, X_train3, s0, c0), y_train = self.__data_generation(input_indexes, output_indexes)        

		return (X_train1, X_train2, X_train3, s0, c0), (y_train, [], []) # pass empty training outputs to extract extract attentions

	def on_epoch_end(self):
		# set length of indexes for each epoch
		self.input_indexes = np.arange(self.xlen)
		self.output_indexes = np.arange(self.ylen)
 
		if self.shuffle == True:
			np.random.shuffle(self.input_indexes)

	def __to_sequence(self, x1, x2, x3, y):
		# convert timeseries batch in sequences
		input_start, output_start = 0, 0

		seqX1, seqX2, seqX3, seqY = [], [], [], []

		while (input_start + input_seq_size) <= len(x1):
			# offset handled during pre-processing
			input_end = input_start + input_seq_size
			output_end = output_start + output_seq_size

			# inputs
			seqX1.append(x1[input_start:input_end])
			seqX2.append(x2[input_start:input_end])

			# outputs
			seqX3.append(x3[output_start:output_end])
			seqY.append(y[output_start:output_end])

			input_start += 1
			output_start += 1
            
		seqX1, seqX2, seqX3, seqY = np.array(seqX1), np.array(seqX2), np.array(seqX3), np.array(seqY)

		return seqX1, seqX2, seqX3, seqY

	def __data_generation(self, input_indexes, output_indexes):

		# dataset_name = 'train_set_V5_withtimefeatures_120hrinput_float32.hdf5'
		f = h5py.File(f"./Data/solar/Processed_Data/{dataset_name}", "r")

		X_train1 = f['train_set']['X1_train'][input_indexes]
		X_train2 = f['train_set']['X2_train'][input_indexes]
		X_train3 = f['train_set']['X3_train'][output_indexes]

		y_train = f['train_set']['y_train'][output_indexes]
		# decoder_input = f['train_set']['y_train'][output_indexes]
		f.close()  

		print(y_train.shape)

		# print(X_train1.shape)
		# av = np.average(X_train1, axis=(1,2))
		# print(av.shape)
		# plt.plot(av[:,0].flatten())
		# plt.plot(y_train[:,0].flatten())
		# plt.show()
		# # print(y_train.shape)
		# exit()



        # convert to sequence data
		X_train1, X_train2, X_train3, y_train = self.__to_sequence(X_train1, X_train2, X_train3, y_train)

		# print(X_train1.shape)
		# av = np.average(X_train1, axis=(2,3))
		# print(av.shape)
		# plt.plot(y_train[:,:,0].flatten())
		# plt.show()
		# exit()
		

		s0 = np.zeros((self.batch_size, n_s))
		c0 = np.zeros((self.batch_size, n_s))

		# print(X_train1.shape)
		# print(X_train2.shape)
		# print(X_train.shape)
		# print(X_train2.shape)

     
		return (X_train1, X_train2, X_train3, s0, c0), y_train


# dataset_name = 'train_set_V5_withtimefeatures_120hrinput_float32.hdf5'


testing_generator = DataGenerator(dataset_name = dataset_name, x_length = x_len, y_length = y_len,  **params)

# idx = 500

# dataset_name = 'train_set_V5_withtimefeatures_120hrinput_float32.hdf5'
# f = h5py.File(f"./Data/wind/Processed_Data/{dataset_name}", "r")
# input1 = f['train_set']['X1_train'][:idx]
# input2 = f['train_set']['X2_train'][:idx]

# input3 = f['train_set']['X3_train'][:idx]
# outputs = f['train_set']['y_train'][:idx]

# 
# s_state0 = np.zeros((idx, ns))
# c_state0 = np.zeros((idx, ns))


# print(input3.shape)



model = load_model(f'./Models/solar_models/q_0.5/solarGeneration_forecast_MainModel_Q_0.5.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention})

print(model.summary())

out = model.predict(testing_generator)

predictions = out[0]

print(predictions.shape)
 
idx = 10
plt.plot(predictions[idx,:,0].flatten(), label="first")
plt.plot(np.insert(predictions[idx+1,:,0],0,0).flatten(), label="second")
plt.plot(predictions[idx+48,:,0].flatten(), label="third")
plt.plot(predictions[idx+96,:,0].flatten(), label="fourth")
plt.legend()



# f['train_set']['y_train'][:idx]
# plt.plot(outputs[:,0])
plt.show()

exit()



# ref = 0

# start_ref = ref*48
# end_ref = 	(ref+1)*48

# dataset_name = 'train_set_V3_withtimefeatures_96hrinput_float32.hdf5'
# f = h5py.File(f"./Data/wind/Processed_Data/{dataset_name}", "r")
# input1 = f['train_set']['X1_train'][ref:ref+1]
# input2 = f['train_set']['X2_train'][ref:ref+1]

# input3 = f['train_set']['X3_train'][start_ref:end_ref]
# outputs = f['train_set']['y_train'][start_ref:end_ref]


# model = load_model(f'./Models/wind_models/q_0.5/windGeneration_forecast_MainModel_Q_0.5.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention})
# # model2 = load_model(f'./Models/wind_models/q_0.9/windGeneration_forecast_MainModel_Q_0.9.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention})
# enoder_temporal_model = load_model(f'./Models/wind_models/q_0.5/windGeneration_encoderModelTemporal_Q_0.5.h5')
# enoder_spatial_model = load_model(f'./Models/wind_models/q_0.5/windGeneration_encoderModelSpatial_Q_0.5.h5')

# # print(model1.summary())
# # # exit()

# # # # # a  = model1.layers[10].get_config()
# # # # # a = model1.get_layer('conv1d_94').get_config()
# # # # # print(K.eval(model1.optimizer.lr))
# # # # # print(a)

# Tx = input2.shape[1]
# Ty = outputs.shape[1]
# height, width, channels = input1.shape[2], input1.shape[3], input1.shape[4]
# times_in_dim = input2.shape[-1]
# times_out_dim = input3.shape[-1]
# n_s = 128

# x_input = Input(shape=(Tx, height, width, channels))
# times_in = Input(shape=(Tx, times_in_dim))
# times_out = Input(shape=(Ty, times_out_dim))
# s_state0 = Input(shape=(n_s,))
# c_state0 = Input(shape=(n_s,))
# dec_inp = Input(shape=(None, 1))

# s_state = s_state0
# c_state = c_state0


def inference_model():

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

# decoder_model = inference_model()



# # idx = 100
# # # # # # # # # inputs = np.average(test_set['X1_test'], axis=(2,3))
# # dataset_name = 'train_set_V11_withtimefeatures_120hrinput.hdf5'
# # f = h5py.File(f"./Data/solar/Processed_Data/{dataset_name}", "r")
# # input1 = f['train_set']['X1_train'][-2:-1]
# # input2 = f['train_set']['X2_train'][-2:-1]

# # input3 = f['train_set']['X3_train'][-48:]
# # outputs = f['train_set']['y_train'][-48:]

# y_prev_int = np.average(input1, axis=(2,3))

# print(outputs.shape)

# predictions = []


# enc_temp_out, encoder_states = enoder_temporal_model.predict([input1, input2])
# enc_spat_out = enoder_spatial_model.predict(input1)

# s_state, c_state = encoder_states[0], encoder_states[1]


# for idx in range(len(outputs)):

# 	if idx == 0:
# 		y_prev =  y_prev_int[:, -48, 0]
# 		y_prev = np.expand_dims(y_prev, axis=1)
# 		y_prev = np.expand_dims(y_prev, axis=-1)
# 	# else:
# 	# 	y_prev =  outputs[idx-1, :]
# 	# 	y_prev = np.expand_dims(y_prev, axis=1)
# 	# 	y_prev = np.expand_dims(y_prev, axis=1)

# 	times_out_single = np.expand_dims(np.expand_dims(input3[idx, :], axis=0),axis=1)

# 	# out = model.predict([input1, input2, np.expand_dims(np.expand_dims(input3[idx, :], axis=0),axis=1), y_prev])
# 	prediction, s_state, c_state, temporal_attention, spatial_attention = decoder_model.predict([y_prev, times_out_single, enc_temp_out, enc_spat_out, s_state, c_state])

# 	# pred = prediction
# 	y_prev = prediction
# 	predictions.append(prediction)


# predictions = np.array(predictions)


# print(predictions.shape)


# plt.plot(predictions[:,0,0,0])
# plt.plot(outputs[:,0])
# plt.show()

# exit()










# Ty = 48



# dataset_name = 'train_set_V2_withtimefeatures_96hrinput_float32.hdf5'
# f = h5py.File(f"./Data/wind/Processed_Data/{dataset_name}", "r")

# model = load_model(f'./Models/wind_models/q_0.5/windGeneration_forecast_MainModel_Q_0.5.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention})
# b = 1000

# X_train1 = np.expand_dims(f['train_set']['X1_train'][b], axis=0)
# X_train2 = np.expand_dims(f['train_set']['X2_train'][b], axis=0)
# X_train3 = np.expand_dims(f['train_set']['X3_train'][b], axis=0)
# y_train = np.expand_dims(f['train_set']['y_train'][b], axis=0)     


# outputs = []

# for t in range(Ty):
# 	if t == 0:
# 		y_prev = np.average(X_train1, axis=(2,3))
# 		y_prev = np.expand_dims(y_prev[:, -1, 0], axis=1)
# 		y_prev = np.expand_dims(y_prev, axis=-1)
# 	else:
# 		y_prev = y_train[:, t-1, :]
# 		y_prev = np.expand_dims(y_prev, axis=1)

# 	output = model.predict([X_train1, X_train2, np.expand_dims(X_train3[:, t, :], axis=1), y_prev])
# 	output = output[0]

# 	outputs.append(output)


# outputs = np.array(outputs)
# print(outputs.shape)

# plt.plot(outputs[:, :].flatten(), color='tab:red', linestyle='--')
# plt.plot(f['train_set']['y_train'][b, :, 0].flatten(), 'k')
# plt.show()

def correlation_analysis(X, Y):

	rs = np.empty((X.shape[0], 1))
	#caclulate 'R^2' for each feature - average over all days
	for l in range(X.shape[0]):
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X[l,:,0], Y[l,:,0])
		rs[l, 0] =r_value**2
		

	print('mean' + '\n R**2: %s' %rs.mean())
	print('max' + '\n R**2: %s' %rs.max())
	print('min' + '\n R**2: %s' %rs.min())

	#get best
	best_fit = np.argmax(rs, axis=0)
	worst_fit = np.argmin(rs, axis=0)
	print(best_fit)
	print(worst_fit)
	# print(X[best_fit,:,0])

	return 




# exit()
idx = 10
set_type = "test"

dataset_name = 'train_set_V2_withtimefeatures_96hrinput_float32.hdf5'
f = h5py.File(f"./Data/wind/Processed_Data/{dataset_name}", "r")
input1 = f[f'{set_type}_set'][f'X1_{set_type}'][:idx]
input2 = f[f'{set_type}_set'][f'X2_{set_type}'][:idx]

input3 = f[f'{set_type}_set'][f'X3_{set_type}'][:idx]
outputs = f[f'{set_type}_set'][f'y_{set_type}'][:idx]






# ns = 128
# s_state0 = np.zeros((idx, ns))
# c_state0 = np.zeros((idx, ns))


# # model1 = load_model(f'./Models/wind_models/q_0.1/windGeneration_forecast_MainModel_Q_0.1.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention})
# model2 = load_model(f'./Models/wind_models/q_0.5/windGeneration_forecast_MainModel_Q_0.5.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention})
# # model3 = load_model(f'./Models/wind_models/q_0.9/windGeneration_forecast_MainModel_Q_0.9.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention})


# # output1 = model1.predict([input1, input2, input3, s_state0, c_state0, outputs])
# output2 = model2.predict([input1, input2, input3, s_state0, c_state0, outputs])
# # output3 = model3.predict([input1, input2, input3, s_state0, c_state0, outputs])


# # test1 = output1[0]
# test2 = output2[0]
# # test3 = output3[0]



# idx = 2
# # plt.plot(test1[idx:idx+7].flatten(), color='tab:red', linestyle='--')
# plt.plot(test2[idx:idx+7].flatten(), color='tab:red', linestyle='--')
# # plt.plot(test3[idx:idx+7].flatten(), color='tab:red', linestyle='--')

# plt.plot(f[f'{set_type}_set'][f'y_{set_type}'][idx:idx+7, :, 0].flatten(), 'k')

# plt.show()


# print(test2.shape)

# correlation_analysis(test2, f[f'{set_type}_set'][f'y_{set_type}'][:, :, 0:1])



# exit()








# TESTING INPUTS & OUTPUTS
# inputs_load = open('./Data/solar/Processed_Data/train_set_V10_withtimefeatures_120hrinput_floar32.hdf5', "r")
# test_set = load(inputs_load)
# inputs_load.close()


# TESTING TIMES
# test_set_load = open("./Data/solar/Processed_Data/time_refsv_V3_withtimefeatures_96hrinput.pkl", "rb") 
# times = load(test_set_load)
# test_set_load.close()


# load sample data
# predictions_load = open("./Data/solar/Processed_Data/predictions.pkl", "rb") 
# predictions = load(predictions_load)
# predictions_load.close()











# idx = 870
# # plt.plot(predictions['quantile_0.1'][idx:idx+7, :, 0].flatten(), color='tab:red', linestyle='--')
# plt.plot(predictions['quantile_0.5'][idx:idx+7, :, 0].flatten(), color='tab:red', linestyle='--')
# # # # plt.plot(predictions['quantile_0.9'][idx:idx+7, :, 0].flatten(), color='tab:red', linestyle='--')
# plt.plot(test_set['y_train'][idx:idx+7, :, 0].flatten(), 'k')
# plt.show()



# exit()
# ###############################################################################################################################
# see train loss
def QuantileLoss(perc, delta=1e-4):
    perc = np.array(perc).reshape(-1)
    perc.sort()
    perc = perc.reshape(1, -1)
    def _qloss(y, pred):
        I = tf.cast(y <= pred, tf.float32)
        d = K.abs(y - pred)
        correction = I * (1 - perc) + (1 - I) * perc
        # huber loss
        huber_loss = K.sum(correction * tf.where(d <= delta, 0.5 * d ** 2 / delta, d - 0.5 * delta), -1)
        print(huber_loss.shape)
        # order loss
        q_order_loss = 0
        return huber_loss + q_order_loss
    return _qloss

perc_points = [0.01, 0.25, 0.5, 0.75, 0.99]

# # # model = load_model(f'./Models/solar_models/quantile_all/solarGeneration_forecast_MainModel_test_Q_all.h5', custom_objects = {'_qloss': QuantileLoss(perc_points), 'attention': attention})
# # model = load_model(f'./Models/wind_models/q_0.1/windGeneration_forecast_MainModel_Q_0.1.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention})
# model1 = load_model(f'./Models/wind_models/q_0.5/windGeneration_forecast_MainModel_Q_0.5.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention})
# # model2 = load_model(f'./Models/wind_models/q_0.9/windGeneration_forecast_MainModel_Q_0.9.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention})

# # print(model1.summary())
# # # exit()

# # # # # a  = model1.layers[10].get_config()
# # # # # a = model1.get_layer('conv1d_94').get_config()
# # # # # print(K.eval(model1.optimizer.lr))
# # # # # print(a)





# idx = 64
# # # # # # # # # inputs = np.average(test_set['X1_test'], axis=(2,3))
# dataset_name = 'train_set_V2_withtimefeatures_96hrinput_float32.hdf5'
# f = h5py.File(f"./Data/wind/Processed_Data/{dataset_name}", "r")
# input1 = f['train_set']['X1_train'][:idx]
# input2 = f['train_set']['X2_train'][:idx]
# input3 = f['train_set']['X3_train'][:idx]
# outputs = f['train_set']['y_train'][:idx]
# # # # # # # # print(outputs.shape)


# # input1 = np.expand_dims(input1, axis=0)
# # input2 = np.expand_dims(input2, axis=0)
# # input3 = np.expand_dims(input3, axis=0)
# # outputs = np.expand_dims(outputs, axis=0)



# # # # # # print(input2.shape)
# # # # # # plt.plot(input2[100:105, :, 0].flatten())
# # # # # # plt.plot(input2[100:105, :, 1].flatten())
# # # # # # plt.plot(input2[100:105, :, 2].flatten())
# # # # # # plt.plot(input2[100:105, :, 3].flatten())
# # # # # # plt.show()

# # # # # # plt.plot(input3[100:105, :, 0].flatten())
# # # # # # plt.plot(input3[100:105, :, 1].flatten())
# # # # # # plt.plot(input3[100:105, :, 2].flatten())
# # # # # # plt.plot(input3[100:105, :, 3].flatten())
# # # # # # plt.show()



# # # # # # exit()
# n_s = 128
# s0 = np.zeros((1, n_s))
# c0 = np.zeros((1, n_s))



# # # # # # # times = times['output_times_test']
# # # # # # # X_test = [test_set['X1_test'], s0, c0]
# # # # # # # ytrue = test_set['y_test']
# # # # # # ytrue = outputs

# # # # # # performance against training datai
# # inputs = [input1, input2, input3, s0, c0, outputs]


# # testing = model.predict(inputs)
# # testing1 = model1.predict(inputs)
# # testing2 = model2.predict(inputs)


# model_out2 = []

# # model_out1 = testing[0]
# for i in range(idx):
# 	print(i)
# 	in1 = np.expand_dims(input1[i], axis=0)
# 	in2 = np.expand_dims(input2[i], axis=0)
# 	in3 = np.expand_dims(input3[i], axis=0)
# 	outs = np.expand_dims(outputs[i], axis=0)

# 	inputs = [in1, in2, in3, s0, c0, outs]

# 	testing1 = model1.predict(inputs)
# 	pred2 = testing1[0]
# 	model_out2.append(pred2)
# # model_out3 = testing2[0]

# model_out2 = np.array(model_out2)




# idx = 115
# # plt.plot(model_out1[idx:idx+7, :, :].flatten(), color='tab:red', linestyle='--')
# plt.plot(model_out2[idx:idx+7, :, :].flatten(), color='tab:red', linestyle='--')
# # plt.plot(model_out3[idx:idx+7, :, :].flatten(), color='tab:red', linestyle='--')
# plt.plot(f['train_set']['y_train'][idx:idx+7, :, 0].flatten(), 'k')
# plt.show()


# correlation_analysis(model_out2, f['train_set']['y_train'][:])
# exit()
# attentions_temp = testing[1]


# ###############################################################################################################################
# att_w_temp = np.transpose(attentions_temp[idx])
# print(att_w_temp.shape)
# # x = a[idx, :, 0] #show average solar iridiance to guadge temporal attention
# x = np.average(f['train_set']['X1_train'], axis=(2,3))[idx, :]
# y = f['train_set']['y_train'][idx, :, 0]
# y_hat = model_out1[idx]
# Tx = 240
# Ty = 48


#apply sns theme
# sns.set_theme("poster")

#make attention plotting function
def temporal_attention_graph(x, y, att_w_temp):

	fig = plt.figure(figsize=(24, 8))
	gs = gridspec.GridSpec(ncols=90, nrows=100)

	upper_axis = fig.add_subplot(gs[0:20, 10:75])
	left_axis = fig.add_subplot(gs[25:, 0:8])
	atten_axis = fig.add_subplot(gs[25:, 10:])

	upper_axis.plot(x)
	upper_axis.set_xlim([0, Tx])
	upper_axis.set_ylim([0, 0.5])
	upper_axis.set_xticks(range(0, Tx))
	upper_axis.set_xticklabels(range(0, Tx))

	left_axis.plot(y, range(0,Ty), label='Prediction')
	left_axis.plot(y_hat, range(0,Ty), label='True')
	left_axis.set_ylim([0, Ty])
	left_axis.set_yticks(range(0, Ty, 4))
	left_axis.set_yticklabels(range(0, Ty, 4))
	left_axis.invert_yaxis()

	sns.heatmap(att_w_temp, cmap='flare', ax = atten_axis)
	atten_axis.set_xticks(range(0, Tx))
	atten_axis.set_xticklabels(range(0, Tx))
	atten_axis.set_yticks(range(0, Ty, 4))
	atten_axis.set_yticklabels(range(0, Ty, 4))

	plt.show()


# temporal_attention_graph(x, y, att_w_temp)
# exit()
# ###############################################################################################################################


# idx = 950

# print(times['output_times_train'][idx])

# plt.plot(predictions['quantile_0.1'][idx, :, 0], 'b')
# plt.plot(predictions['quantile_0.5'][idx, :, 0], 'r')
# plt.plot(predictions['quantile_0.9'][idx, :, 0], 'g')
# # plt.plot(predictions['quantile_0.2'][idx, :, 0], 'k')
# plt.plot(test_set['y_train'][idx, :, 0], 'p')
# plt.show()

# exit()


# helper function to elvaluate 
def probabilistic_accuracy(y_true, lower_pred, upper_pred, alpha):
	'''
	Theory from Bazionis & Georgilakis (2021): https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiUprb39qbyAhXNgVwKHWVsA50QFnoECAMQAQ&url=https%3A%2F%2Fwww.mdpi.com%2F2673-4826%2F2%2F1%2F2%2Fpdf&usg=AOvVaw1AWP-zHuNGrw8pgDfUS09e
	func to caluclate probablistic forecast performance
	Prediction Interval Coverage Probability (PICP)
	Prediction Interval Nominla Coverage (PINC)
	Average Coverage Error (ACE) [PICP - PINC]
	'''
	test_len = len(y_true)
	print(test_len)

	y_true = y_true.ravel()
	lower_pred = lower_pred.ravel()
	upper_pred = upper_pred.ravel()

	# picp_ind = np.sum((y_true > lower_pred) & (y_true <= upper_pred))

	# print(picp_ind)

	picp = (np.sum((y_true > lower_pred) & (y_true <= upper_pred)) / test_len) 

	pinc = 100 * (1 - (alpha))

	ace = picp - pinc # closer to '0' higher the reliability

	r = np.max(y_true) - np.min(y_true)

	# PI normalised width
	pinaw = (1 / (test_len * r)) * np.sum((upper_pred - lower_pred))

	# PI normalised root-mean-sqaure width 
	pinrw = (1/r) * np.sqrt( (1/test_len) * np.sum((upper_pred - lower_pred)**2) )

	# create pandas df
	metrics = pd.DataFrame({'PICP': picp, 'PINC': pinc, 'ACE': ace, 'PINAW': pinaw, 'PINRW': pinrw}, index={alpha})
	metrics.index.name = 'Prediction_Interval'

	print(metrics)

	return metrics





# probabilistic_accuracy(test_set['y_train'][:1000,:,:], predictions['quantile_0.1'], predictions['quantile_0.9'], 0.8)

# idx = 400

# plt.plot(predictions['quantile_0.1'][idx, :, 0], 'b')
# plt.plot(predictions['quantile_0.5'][idx, :, 0], 'r')
# plt.plot(predictions['quantile_0.9'][idx, :, 0], 'g')
# plt.plot(test_set['y_train'][idx, :, 0], 'p')
# plt.show()


# exit()




os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #hide tensorflow error 

np.set_printoptions(threshold=sys.maxsize)

###########################################_____LOAD & PRE-PROCESS DATA_____###########################################

#cache current working directory of main script
workingDir = os.getcwd()

model_directory = workingDir + '/Models/wind_models/'
folders = os.listdir(model_directory)

# create dictionary to store predictions from each model
q_predictions, q_spatial_attentions, q_temporal_attentions = {}, {}, {}


def QuantileLoss(perc, delta=1e-4):
    perc = np.array(perc).reshape(-1)
    perc.sort()
    perc = perc.reshape(1, -1)
    def _qloss(y, pred):
        I = tf.cast(y <= pred, tf.float32)
        d = K.abs(y - pred)
        correction = I * (1 - perc) + (1 - I) * perc
        # huber loss
        huber_loss = K.sum(correction * tf.where(d <= delta, 0.5 * d ** 2 / delta, d - 0.5 * delta), -1)
        print(huber_loss.shape)
        # order loss
        q_order_loss = 0
        return huber_loss + q_order_loss
    return _qloss

perc_points = [0.01, 0.25, 0.5, 0.75, 0.99]





final_predictions = {}

# loop for each qunatile
for folder in folders:
	print(folders)
	quantile = folder[-3:]
	# quantile = 'all'

	print(f'loading {folder} model...')

	# model = load_model(f'./Models/solar_models/{folder}/windGeneration_forecast_MainModel_Q_{quantile}.h5', custom_objects = {'_qloss': QuantileLoss(perc_points), 'attention': attention})
	model = load_model(f'./Models/wind_models/{folder}/windGeneration_forecast_MainModel_Q_{quantile}.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention})
	# print(model.summary())


	enoder_temporal_model = load_model(f'./Models/wind_models/{folder}/windGeneration_encoderModelTemporal_Q_{quantile}.h5')
	enoder_spatial_model = load_model(f'./Models/wind_models/{folder}/windGeneration_encoderModelSpatial_Q_{quantile}.h5')

	# previous predictions and states between predictions
	prev_s_state = None
	prev_c_state = None
	prev_prediction = None


	# load data

	# # TESTING INPUTS & OUTPUTS
	# inputs_load = open('./Data/solar/Processed_Data/train_set_V3_withtimefeatures_96hrinput__.pkl', "rb")
	# test_set = load(inputs_load)
	# inputs_load.close()
	 

	# # TESTING TIMES
	# test_set_load = open("./Data/solar/Processed_Data/time_refsv_V3_withtimefeatures_96hrinput.pkl", "rb") 
	# times = load(test_set_load)
	# test_set_load.close()

	# print(test_set['X1_test'].shape)
	# print(test_set['y_test'].shape)
	# #################################################################################################################################################################################

	Tx = input2.shape[1]
	Ty = outputs.shape[1]
	height, width, channels = input1.shape[2], input1.shape[3], input1.shape[4]
	times_in_dim = input2.shape[-1]
	times_out_dim = input3.shape[-1]
	n_s = 128


	# define inputs
	x_input = Input(shape=(Tx, height, width, channels))
	times_in = Input(shape=(Tx, times_in_dim))
	times_out = Input(shape=(Ty, times_out_dim))
	s_state0 = Input(shape=(n_s,))
	c_state0 = Input(shape=(n_s,))
	dec_inp = Input(shape=(None, 1))

	s_state = s_state0
	c_state = c_state0



	# empty dictionaries for decoder models
	decoder_models, enoder_temporal_models, enoder_spatial_models = {}, {}, {}

	######## model for inference #############
	def inference_model():

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
		dec_input_concat_test = concatenate([context_test, dec_input_test], axis=-1)
		dec_input_concat_test_int = concatenate([context_test, dec_input_test_int], axis=-1)

		# combine with decoder inputs
		dec_input_concat_test = concatenate([dec_input_concat_test, times_out_test], axis=-1)
		dec_input_concat_test_int = concatenate([dec_input_concat_test_int, times_out_test], axis=-1)

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


	# for idx in range(1, Ty+1):
	# 	decoder_models[f'{idx}'] = inference_model(idx)

	decoder_inference = inference_model()

	# print(decoder_models)
	# print(decoder_models[f'{idx}'][0])
	# sys.exit()
	print('inference model made')

	############_________Inference__________##############
	# sample_size = 1000

	for sample in range(input1.shape[0]):
		# print(sample)

		main_input = input1[sample:sample+1]
		times_in = input2[sample:sample+1]
		times_out = input3[sample:sample+1]
		y_train = outputs[sample:sample+1]
		s0 = np.zeros((1, n_s))
		c0 = np.zeros((1, n_s))

		s_state = s0
		c_state = c0
		predictions = []

		# inference run
		enc_temp_out, s_state, c_state  = enoder_temporal_model.predict([main_input, times_in])
		enc_spat_out = enoder_spatial_model.predict(main_input)

		# intial decoder input
		# dec_input_int = enc_temp_out[:,-1,:]
		# dec_input_int = K.expand_dims(dec_input_int, axis=1)
		# dec_input = np.zeros((sample_size, 1, 1))

		# y_prev = np.mean(main_input, axis=(2,3))
		# y_prev = y_prev[:, -48, 0]
		dec_input = enc_temp_out[:,-1,-1]
		dec_input = np.expand_dims(dec_input, axis=1)
		dec_input = np.expand_dims(dec_input, axis=-1)

		# use encoder states or previous prediction
		if prev_prediction == None:
			print('first pass')
		else:
			y_prev = prev_prediction
			# s_state = prev_s_state
			# c_state = prev_c_state

		for t in range(1, Ty+1):

			# get current 'times out' reference
			times_out_single = times_out[:,t-1,:]
			times_out_single = K.expand_dims(times_out_single, axis=1)

		    # dec_output, s_state, c_state = decoder(decoder_input, times_out_single, s_state, c_state)
			prediction, s_state, c_state, temporal_attention, spatial_attention = decoder_inference.predict([dec_input, times_out_single, enc_temp_out, enc_spat_out, s_state, c_state])
			# prediction, s_state, c_state, temporal_attention, spatial_attention = decoder_models[f'{t}'].predict([dec_input, dec_input_int, times_out_single, enc_temp_out, enc_spat_out, s_state, c_state])
			dec_input = outputs[sample:sample+1,t-1,:]
			dec_input =  K.expand_dims(dec_input, axis=1)
			# dec_input = prediction 
			# dec_input = Reshape((1, 1))(dec_input) 

			# print('dec_input')
			print(f"shape:{dec_input.shape}")

			# predictions.append(prediction)
			# folder = 'quantile_0.5'

			if t == 1:
				q_predictions[folder] = prediction
				enoder_temporal_models[folder] = temporal_attention
				q_spatial_attentions[folder] = spatial_attention
			else:
				q_predictions[folder] = np.concatenate([q_predictions[folder], prediction], axis=1)
				enoder_temporal_models[folder] = np.concatenate([enoder_temporal_models[folder], temporal_attention], axis=-1)
				q_spatial_attentions[folder] = np.concatenate([q_spatial_attentions[folder], spatial_attention], axis=-1)

		prev_s_state = s_state
		prev_c_state = c_state
		prev_prediction = prediction

		# print('**************************')
		# print(q_predictions[folder].shape)
		if sample == 0:
			final_predictions[folder] = q_predictions[folder]
		else:
			final_predictions[folder] = np.concatenate([final_predictions[folder], q_predictions[folder]], axis=0)





print(final_predictions.keys())
print(final_predictions[folder].shape)




# exit()
idx = 2
# print(q_predictions[folder][50])
# save some data for debugging
# with open("./Models/solar_models/predictions.pkl", "wb") as predictions:
# 	dump(q_predictions, predictions)


# plt.plot(final_predictions['q_0.1'][idx:idx+5, :, 0].flatten())
plt.plot(final_predictions['q_0.5'][idx:idx+7, :, 0].flatten())
# plt.plot(final_predictions['q_0.9'][idx:idx+5, :, 0].flatten())
plt.plot(outputs[idx:idx+7, :, 0].flatten())
plt.show()


exit()


# probabilistic_accuracy(test_set['y_train'][:sample_size,:,:], q_predictions['quantile_0.1'], q_predictions['quantile_0.9'], 0.8)
# print(q_predictions['quantile_all'].shape)


# plt.plot(q_predictions['quantile_all'][idx, :, :], color='tab:red', linestyle='--')
# plt.plot(test_set['y_train'][idx, :, 0], 'k')
# plt.show()







def correlation_analysis(X, Y):

	rs = np.empty((X.shape[0], 1))
	#caclulate 'R^2' for each feature - average over all days
	for l in range(X.shape[0]):
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X[l,:,0], Y[l,:,0])
		rs[l, 0] =r_value**2
		

	print('mean' + '\n R**2: %s' %rs.mean())
	print('max' + '\n R**2: %s' %rs.max())
	print('min' + '\n R**2: %s' %rs.min())

	#get best
	best_fit = np.argmax(rs, axis=0)
	worst_fit = np.argmin(rs, axis=0)
	print(best_fit)
	print(worst_fit)
	# print(X[best_fit,:,0])

	return 


correlation_analysis(q_predictions['quantile_0.5'], test_set['y_test'][:100])


exit()


#function for plotting temporal attention
def temporal_attention_graph(x, y, y_hat, att_w_temp):

	fig = plt.figure(figsize=(24, 8))
	gs = gridspec.GridSpec(ncols=90, nrows=100)

	upper_axis = fig.add_subplot(gs[0:20, 10:75])
	left_axis = fig.add_subplot(gs[25:, 0:8])
	atten_axis = fig.add_subplot(gs[25:, 10:])

	upper_axis.plot(x)
	upper_axis.set_xlim([0, Tx])
	upper_axis.set_ylim([0, 0.5])
	upper_axis.set_xticks(range(0, Tx))
	upper_axis.set_xticklabels(range(0, Tx))

	left_axis.plot(y, range(0,Ty), label='True')
	left_axis.plot(y_hat, range(0,Ty), label='Prediction')
	left_axis.set_ylim([0, Ty])
	left_axis.set_yticks(range(0, Ty, 4))
	left_axis.set_yticklabels(range(0, Ty, 4))
	left_axis.invert_yaxis()
	left_axis.legend()
    

	sns.heatmap(att_w_temp, cmap='flare', ax = atten_axis)
	atten_axis.set_xticks(range(0, Tx))
	atten_axis.set_xticklabels(range(0, Tx))
	atten_axis.set_yticks(range(0, Ty, 4))
	atten_axis.set_yticklabels(range(0, Ty, 4))

	plt.show()


att_w_temp_test = np.transpose(total_temporal_attn_test[idx])

idx = 884
x = np.average(test_set['X1_train'], axis=(2,3))[idx, :]
y = y_train[idx, :, 0]
# y_hat_train = model_out1[idx]
y_hat_test = predictions_test[idx]

att_w_temp_test = np.transpose(total_temporal_attn_test[idx])

temporal_attention_graph(x, y, y_hat_test, att_w_temp_test)

exit()

# #################################################################################################################################################################################

idx = 468


#load scaler_y 
# scalery_load = open("./Data/solar/scaler_y.pkl", "rb") 
# scaler_y = load(scalery_load)
# scalery_load.close()




# print(test_set.keys())
# print(test_set['X1_test'].shape)

# print(test_set['X2_test'].shape)
# print(test_set['X3_test'].shape)



# inputs = np.average(test_set['X1_test'], axis=(2,3))
input1 = test_set['X1_train']
input2 = test_set['X2_train']
input3 = test_set['X3_train']
outputs = test_set['y_train']
print(outputs.shape)



n_s = 64
# s0 = np.zeros((test_set['X1_test'].shape[0], n_s))
# c0 = np.zeros((test_set['X1_test'].shape[0], n_s))

# 264
s0 = np.zeros((2380, n_s))
c0 = np.zeros((2380, n_s))

# times = times['output_times_test']
# X_test = [test_set['X1_test'], s0, c0]
# ytrue = test_set['y_test']
ytrue = outputs

# performance against training data
inputs = [input1, input2, input3, s0, c0, outputs]






# print('predicitng...')
# yhat1 = model.predict(inputs)
# model_output = yhat1[0]


# layers = [layer.name for layer in model.layers]
# weights = [layer.get_weights() for layer in model.layers]
# model = Model(inputs=model.input, outputs=[model.output, model.get_layer('attentions_47').output])


#load scaler_y 
scalery_load = open("./Data/solar/scaler_y.pkl", "rb") 
scaler_y = load(scalery_load)
scalery_load.close()



print('predicitng...')
yhat1 = model.predict(inputs)
model_out1 = yhat1[0]
attentions_temp = yhat1[1]
attentions_spat = yhat1[2]


# a  =pd.DataFrame(attentions_temp[idx])#.to_csv("attentions.csv")


# df = pd.DataFrame(data=a.values,index=a.index, columns=a.columns).stack()

# df.to_csv("attentions.csv")




# evaluate model
# print(f'evaluation:')


# plot spatial attention
# att_w_spat = np.transpose(attentions_spat[idx])
# print(att_w_spat.shape)
# print(attentions_temp.shape)


def spatial_attention_graph(att_w_spat):

	fig = plt.figure(figsize=(16, 20))
	print(att_w_spat.shape)


	for t in range(att_w_spat.shape[0]):
		print(t)
		# resize back to rect 
		attention = np.resize(att_w_spat[t], (16,20))
		print(attention.shape)
		grid_size = max(np.ceil(len(att_w_spat[1])/2), 2)
		ax = fig.add_subplot(7, 7, t+1)
		im = ax.imshow(attention, cmap='viridis')
		# sns.heatmap(att_w_temp, cmap='flare', ax = atten_axis)
		ax.set_xticks = []
		ax.set_yticks = []
		ax.set_title(f'testing{t}')
		# fig.colorbar(ax=ax)


	plt.tight_layout()
	fig.colorbar(im, orientation='horizontal', fraction=.1)
	plt.show()



# spatial_attention_graph(att_w_spat)
# ############################################################################################################################################################
# ############################################################################################################################################################
# ############################################################################################################################################################



def spatial_vis(spatial_data, title, height_scale, width_scale, frame_num):

	fig = plt.figure(figsize=[8,10])  # a new figure window
	ax_set = fig.add_subplot(1, 1, 1)

	# create baseline map
	# spatial data on UK basemap
	df = pd.DataFrame({
		'LAT': [49.78, 61.03],
		'LON': [-11.95, 1.55],
	})

	geo_df = geopandas.GeoDataFrame(df, crs = {'init': 'epsg:4326'}, 
			geometry=geopandas.points_from_xy(df.LON, df.LAT)).to_crs(epsg=3857)

	ax = geo_df.plot(
		figsize= (8,10),
		alpha = 0,
		ax=ax_set,
	)

	plt.title(title)
	ax.set_axis_off()

	# add basemap
	url = 'http://tile.stamen.com/terrain/{z}/{x}/{y}.png'
	zoom = 10
	xmin, xmax, ymin, ymax = ax.axis()
	basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url)
	ax.imshow(basemap, extent=extent, interpolation='gaussian')
	attn_over = np.resize(spatial_data[0], (height_scale, width_scale))
	
	gb_shape = geopandas.read_file("./Data/shapefiles/GBR_adm/GBR_adm0.shp").to_crs(epsg=3857)
	irl_shape = geopandas.read_file("./Data/shapefiles/IRL_adm/IRL_adm0.shp").to_crs(epsg=3857)
	gb_shape.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5, alpha=0.4)
	irl_shape.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5, alpha=0.4)
	overlay = ax.imshow(attn_over, cmap='viridis', alpha=0.5, extent=extent)
	# ax.axis((xmin, xmax, ymin, ymax))
	txt  = fig.text(.5, 0.09, '', ha='center')

	
	def update(i):
		# print(i)
		
		spatial_over = np.resize(spatial_data[i], (height_scale, width_scale))
		# overlay = ax.imshow(spatial_over, cmap='viridis', alpha=0.5, extent=extent)
		overlay.set_data(spatial_over)
		txt.set_text(f"Timestep: {i}")
		# plt.cla()

		return [overlay, txt]


	animation_ = FuncAnimation(fig, update, frames=frame_num, blit=False, repeat=False)
	plt.show(block=True)	
	# animation_.save(f'{title}_animation.gif', writer='imagemagick')


# spatial_vis(test_set['X1_test'][idx, :, :, :, 1], 'Cloud Cover (input)', test_set['X1_test'].shape[2], test_set['X1_test'].shape[3], 96)
# spatial_vis(att_w_spat, 'Spatial Context', 16, 20, 48)


# ############################################################################################################################################################
# ############################################################################################################################################################
# ############################################################################################################################################################
# create script for testing model


# load test model
# model = load_model('./Models/solarGeneration_forecast_Q_0.5.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention}, compile=False)
# model_test = load_model('./Models/solarGeneration_forecast_test2_Q_0.5.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention}, compile=False)
# # overwrite weights



# weights = model_test.get_weights()
# print(weights[26].shape)
# print(weights[27].shape)
# exit()

# for weight in weights:
# 	print(weight.shape)
# exit()


# model_test.load_weights('./Models/solarGeneration_forecast_test2_Q_0.5.h5')






# exit()










# ############################################################################################################################################################
# ############################################################################################################################################################
# ############################################################################################################################################################





# create helper function for basemap
# def add_basemap(i, ax, zoom, url='http://tile.stamen.com/terrain/{z}/{x}/{y}.png'):
# 	xmin, xmax, ymin, ymax = ax.axis()
# 	basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url)
# 	ax.imshow(basemap, extent=extent, interpolation='bilinear')
# 	# attn_over = np.resize(att_w_spat[i], (16,20))
# 	# ax.imshow(attn_over, cmap='viridis', alpha=0.7, extent=extent)
# 	gb_shape = geopandas.read_file("./Data/shapefiles/GBR_adm/GBR_adm0.shp").to_crs(epsg=3857)
# 	irl_shape = geopandas.read_file("./Data/shapefiles/IRL_adm/IRL_adm0.shp").to_crs(epsg=3857)
# 	gb_shape.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5, alpha=0.4)
# 	irl_shape.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5, alpha=0.4)

# 	# restore original x/y limits
# 	ax.axis((xmin, xmax, ymin, ymax))



# # def attention_update(ax, ):
# # 	extent = ax.get_window_extent()
# # 	ax.imshow(attn_over, cmap='viridis', alpha=0.7, extent=extent)


# # spatial data on UK basemap
# df = pd.DataFrame({
# 	'LAT': [49.78, 61.03],
# 	'LON': [-11.95, 1.55],
# })

# geo_df = geopandas.GeoDataFrame(df, crs = {'init': 'epsg:4326'}, 
# 		geometry=geopandas.points_from_xy(df.LON, df.LAT)).to_crs(epsg=3857)

# ax = geo_df.plot(
# 	figsize= (16,20),
# 	alpha = 1,
# 	ax=ax_set
# )


# # add base map the the image
# # gb_shape = geopandas.read_file("./Data/shapefiles/GBR_adm/GBR_adm0.shp").to_crs(epsg=3857)
# # irl_shape = geopandas.read_file("./Data/shapefiles/IRL_adm/IRL_adm0.shp").to_crs(epsg=3857)
# # gb_shape.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5, alpha=0.4)
# # irl_shape.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5, alpha=0.4)

# # add_basemap(ax, zoom=10)

# def animate(i, ax):
# 	add_basemap(i, ax, zoom=10)






# ############################################################################################################################################################
# ############################################################################################################################################################
# ############################################################################################################################################################

# print('predicitng...')
# yhat1 = model.predict(X_test)
# yhat2 = model2.predict(X_test)
# yhat3 = model1.predict(X_test)
# yhat4 = model2.predict(X_test)

print(attentions_temp.shape)
print(model_out1.shape)
print(ytrue.shape)

# yhat1 = scaler_y.inverse_transform(yhat1)


#print some test data
a = np.mean(np.mean(test_set['X1_train'],axis=2),axis=2)

# for i in range(a.shape[0]):
# 	plt.plot(a[i,:,0])
# plt.show()
	#
# print(times[25])

plt.plot(a[24,:,0])
# plt.plot(a[100,:,1])
# plt.plot(a[100,:,2])
plt.show()

# print(yhat1)

# print(yhat.shape)
# print(y_test.shape)

print(model_out1[20,:])


# # plt.plot(y_test[5,:,:])
# plt.plot(model_output[1216,:])
# # plt.plot(yhat2[80,:])
# plt.plot(ytrue[1216,:])

# plt.legend()
# plt.show()



#function to preform stats analysis
# def correlation_analysis(X, Y):

# 	rs = np.empty((X.shape[0], 1))
# 	#caclulate 'R^2' for each feature - average over all days
# 	for l in range(X.shape[0]):
# 		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X[l,:,0], Y[l,:,0])
# 		rs[l, 0] =r_value**2
		

# 	print('mean' + '\n R**2: %s' %rs.mean())
# 	print('max' + '\n R**2: %s' %rs.max())
# 	print('min' + '\n R**2: %s' %rs.min())

# 	#get best
# 	best_fit = np.argmax(rs, axis=0)
# 	worst_fit = np.argmin(rs, axis=0)
# 	print(best_fit)
# 	print(worst_fit)
# 	# print(X[best_fit,:,0])

# 	return 


# correlation_analysis(model_out1, ytrue)

Tx = 96
Ty = 48


#load train data
# train_set_load = open("./Data/solar/Processed_Data/train_setv8.pkl", "rb") 
# train_set = load(train_set_load)
# train_set_load.close()




att_w_temp = np.transpose(attentions_temp[idx])
print(att_w_temp.shape)
# x = a[idx, :, 0] #show average solar iridiance to guadge temporal attention
x = np.average(test_set['X1_train'], axis=(2,3))[idx, :]
y = ytrue[idx, :, 0]
y_hat = model_out1[idx]

#apply sns theme
# sns.set_theme("poster")

#make attention plotting function
def temporal_attention_graph(x, y, att_w_temp):

	fig = plt.figure(figsize=(24, 8))
	gs = gridspec.GridSpec(ncols=90, nrows=100)

	upper_axis = fig.add_subplot(gs[0:20, 10:75])
	left_axis = fig.add_subplot(gs[25:, 0:8])
	atten_axis = fig.add_subplot(gs[25:, 10:])

	upper_axis.plot(x)
	upper_axis.set_xlim([0, Tx])
	upper_axis.set_ylim([0, 0.5])
	upper_axis.set_xticks(range(0, Tx))
	upper_axis.set_xticklabels(range(0, Tx))

	left_axis.plot(y, range(0,Ty), label='Prediction')
	left_axis.plot(y_hat, range(0,Ty), label='True')
	left_axis.set_ylim([0, Ty])
	left_axis.set_yticks(range(0, Ty, 4))
	left_axis.set_yticklabels(range(0, Ty, 4))
	left_axis.invert_yaxis()

	sns.heatmap(att_w_temp, cmap='flare', ax = atten_axis)
	atten_axis.set_xticks(range(0, Tx))
	atten_axis.set_xticklabels(range(0, Tx))
	atten_axis.set_yticks(range(0, Ty, 4))
	atten_axis.set_yticklabels(range(0, Ty, 4))

	plt.show()





# input_data = {'solar': x[:,0], 'cloud_cover': x[:,1]}
# inputs = pd.DataFrame(input_data)
# output = pd.DataFrame(y)

# inputs.to_csv('inputs.csv')
# output.to_csv('output.csv')
# exit()





temporal_attention_graph(x, y, att_w_temp)










# def inference_model(idx):

# 	# LSTM Encoder
# 	enc_model_temp_test = Model(x_input, lstm_enc_output)
# 	# CNN Encoder
# 	enc_model_spat_test = Model(x_input, ccn_enc_output)

# 	# Encoder outputs for setup
# 	ccn_enc_output_test = Input(shape=(320, 256))
# 	lstm_enc_output_test = Input(shape=(Tx, n_s))

# 	# Decoder Input
# 	dec_input_test = Input(shape=(1, 1))
# 	dec_input_test_int = Input(shape=(1, 64))
# 	times_out_test = Input(shape=(1, times_out_dim))
# 	# idx = Input(shape=(1))

# 	# context and previous output
# 	attn_weights_temp_test, context_temp_test = temporal_attn(lstm_enc_output_test, s_state0, c_state0)
# 	attn_weights_spat_test, context_spat_test = spatial_attn(ccn_enc_output_test, s_state0, c_state0)

# 	# context & previous output combine
# 	context_test = concatenate([context_spat_test, context_temp_test], axis=-1) 
# 	dec_input_concat_test = concatenate([context_test, dec_input_test], axis=-1)
# 	dec_input_concat_test_int = concatenate([context_test, dec_input_test_int], axis=-1)

# 	# combine with decoder inputs
# 	dec_input_concat_test = concatenate([dec_input_concat_test, times_out_test], axis=-1)
# 	dec_input_concat_test_int = concatenate([dec_input_concat_test_int, times_out_test], axis=-1)

# 	# Decoder inference
# 	if idx == 1:
# 		dec_output, s_state, c_state = model.get_layer('lstm_1')(dec_input_concat_test_int, initial_state=[s_state0, c_state0])
# 	else:   
# 		dec_output, s_state, c_state = model.get_layer(f'lstm_{idx}')(dec_input_concat_test, initial_state=[s_state0, c_state0])

# 	pred_test = predict(dec_output)

# 	# Inference Model
# 	model = Model(inputs=[dec_input_test, dec_input_test_int, times_out_test, lstm_enc_output_test, ccn_enc_output_test, s_state0, c_state0], outputs=[pred_test, s_state, c_state, attn_weights_temp_test, attn_weights_spat_test])  
	     

# 	return model


# for idx in range(1, Ty+1):
# 	decoder_models[f'{idx}'] = inference_model(idx)










# for q in quantiles:
# 	# make unique folder for each folder
# 	os.mkdir(f'/content/drive/My Drive/quantile_{q}')
# 	# check if central estimate model exists
# 	if os.path.exists('./Models/solar_models/quantile_0.5/solarGeneration_forecast_MainModel_Q_0.5.h5'):
# 		model = load_model('./Models/solar_models/quantile_0.5/solarGeneration_forecast_MainModel_Q_0.5.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention})
# 		# re-compile model with new loss for current quantile 
# 		model.compile(loss = [lambda y,f: defined_loss(q,y,f), None, None], optimizer= optimizer, metrics = ['mae'])
# 		# load trained weights from q_0.5
# 		model.load_weights(f'/content/drive/My Drive/solarGeneration_forecast_weights_Q_0.5.h5')
# 		# freeze all weights apart from decoder fro re-train
# 		model = freeze_decoder_train(model)
# 		model.fit(training_generator, epochs = 20)

# 	else:
# 		# train median model for reference to other qunatiles
# 		model.compile(loss = [lambda y,f: defined_loss(q,y,f), None, None], optimizer= optimizer, metrics = ['mae'])
# 		model.fit(training_generator, epochs = 20)
# 		# save wegihts 
# 		model.save_weights(f'/content/drive/My Drive/solarGeneration_forecast_weights_Q_{q}.h5')
# 		# save some additional models for inference
# 		enc_model_temp_test = Model(inputs = [x_input, times_in], outputs=[lstm_enc_output])
# 		enc_model_spat_test = Model(x_input, ccn_enc_output)
# 		enc_model_temp_test.save(f'/content/drive/My Drive/quantile_{q}/solarGeneration_encoderModelTemporal_Q_{q}.h5')
# 		enc_model_spat_test.save(f'/content/drive/My Drive//quantile_{q}/solarGeneration_encoderModelSpatial_Q_{q}.h5')


# 	model.save(f'/content/drive/My Drive/quantile_{q}/solarGeneration_forecast_MainModel_Q_{q}.h5')









