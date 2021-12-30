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






type ="wind"

if type == 'wind':
	dataset_name = 'train_set_V11_avglabels.hdf5'
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
output_seq_size = 1
n_s = 128


params = {'batch_size': 1,
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
		# return int(np.floor((self.xlen - (input_seq_size-1)) / self.batch_size))
		return int(np.floor((self.ylen - input_seq_size - (output_seq_size-1)) / self.batch_size))

	def __getitem__(self, index):

		# print(index)        

		# input_indexes = self.input_indexes[(index*self.batch_size) : (index*self.batch_size)+ (self.batch_size + (input_seq_size-1))]
		# output_indexes = self.output_indexes[(index*self.batch_size) : (index*self.batch_size) + (self.batch_size + (output_seq_size-1))]

		input_indexes = self.input_indexes[(index*self.batch_size) : (index*self.batch_size) + (self.batch_size + (input_seq_size-1))]
		output_indexes = self.output_indexes[(index*self.batch_size) + input_seq_size : (index*self.batch_size) + input_seq_size + (self.batch_size + (output_seq_size-1))]

		# self.index_ref_in += self.batch_size + (input_seq_size-1)
		# self.index_ref_out += self.batch_size + (output_seq_size-1)

		# Generate data
		(X_train1, X_train2, X_train3, X_train4), y_train = self.__data_generation(input_indexes, output_indexes)        

		return (X_train1, X_train2, X_train3, X_train4), (y_train, [], []) # pass empty training outputs to extract extract attentions

	def on_epoch_end(self):
		# set length of indexes for each epoch
		self.input_indexes = np.arange(self.xlen)
		self.output_indexes = np.arange(self.ylen)
 
		if self.shuffle == True:
			np.random.shuffle(self.input_indexes)

	def to_sequence(self, x1, x2, x3, x4, y):
		# convert timeseries batch in sequences
		input_start, output_start = 0, 0

		seqX1, seqX2, seqX3, seqX4, seqY = [], [], [], [], []

		while (input_start + input_seq_size) <= len(x1):
			# offset handled during pre-processing
			input_end = input_start + input_seq_size
			output_end = output_start + output_seq_size

			# inputs
			seqX1.append(x1[input_start:input_end])
			seqX2.append(x2[input_start:input_end])

			# outputs
			seqX3.append(x3[output_start:output_end])
			seqX4.append(x4[output_start:output_end])
			seqY.append(y[output_start:output_end])

			input_start += 1  
			output_start += 1
            
		seqX1, seqX2, seqX3, seqX4, seqY = np.array(seqX1), np.array(seqX2), np.array(seqX3), np.array(seqX4), np.array(seqY)

		return seqX1, seqX2, seqX3, seqX4, seqY

	def __data_generation(self, input_indexes, output_indexes):

		f = h5py.File(f"./Data/{type}/Processed_Data/{dataset_name}", "r")      
		X_train1 = f['train_set']['X1_train'][input_indexes]
		X_train2 = f['train_set']['X2_train'][input_indexes]
		X_train3 = f['train_set']['X3_train'][output_indexes]
		X_train4 = f['train_set']['X1_train'][output_indexes][:,:,:,1:]
		X_train4 = np.average(X_train4, axis=(1,2))

		# print(X_train4.shape)
		# sys.exit()        

		y_train = f['train_set']['y_train'][output_indexes]
		# decoder_input = f['train_set']['y_train'][output_indexes]
		f.close()  

		# print(X_train1.shape)
		
        # convert to sequence data
		X_train1, X_train2, X_train3, X_train4, y_train = self.to_sequence(X_train1, X_train2, X_train3, X_train4, y_train)

		s0 = np.zeros((self.batch_size, n_s))
		c0 = np.zeros((self.batch_size, n_s))

		# print(X_train1.shape)
		# print(y_train.shape)


		return (X_train1, X_train2, X_train3, X_train4), y_train



training_generator = DataGenerator(dataset_name = dataset_name, x_length = x_len, y_length = y_len,  **params)




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


set_type = 'test'

X_train1 = f[f'{set_type}_set'][f'X1_{set_type}'][0:1000]
X_train2 = f[f'{set_type}_set'][f'X2_{set_type}'][0:1000]
X_train3 = f[f'{set_type}_set'][f'X3_{set_type}'][0:1000]
X_train4 = f[f'{set_type}_set'][f'X1_{set_type}'][0:1000]
y_train = f[f'{set_type}_set'][f'y_{set_type}'][0:1000]


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
print('test data created')
Q = 0.5

model = load_model(f'./Models/{type}_models/q_{Q}/{type}Generation_forecast_MainModel_Q_{Q}.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention, 'Activation': Activation(swish)})
# model1 = load_model(f'./Models/{type}_models/q_0.01/{type}Generation_forecast_MainModel_Q_0.01.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention, 'Activation': Activation(swish)})
model2 = load_model(f'./Models/{type}_models/q_0.9/{type}Generation_forecast_MainModel_Q_0.9.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention, 'Activation': Activation(swish)})
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


enoder_temporal_model = load_model(f'./Models/{type}_models/q_{Q}/{type}Generation_encoderModel_temporal_Q_{Q}.h5')
enoder_spatial_model = load_model(f'./Models/{type}_models/q_{Q}/{type}Generation_encoderModel_spatial_Q_{Q}.h5')


def inference_model():

	# Encoder outputs for setup
	ccn_enc_output_test = Input(shape=(320, n_s)) # 1584 in normal enc/dec
	lstm_enc_output_test = Input(shape=(Tx, n_s)) #+ times_in_dim

	# Decoder Inputs
	nwp_out = Input(shape=(Ty, channels-1))
	times_out = Input(shape=(Ty, times_out_dim))

	# hidden state inputs
	s_state_in = Input(shape=(n_s))
	c_state_in = Input(shape=(n_s))

	attn_weights_spat, context_spat  = model.get_layer('spatial_attention')(ccn_enc_output_test, s_state_in, c_state_in)
	attn_weights_temp, context_temp = model.get_layer('temporal_attention')(lstm_enc_output_test, s_state_in, c_state_in)

	# encoder combine
	encoder_out_combine = concatenate([context_spat, context_temp], axis=-1) 
	# encoder_out_combine = RepeatVector(Ty)(encoder_out_combine)

	# combine with decoder inputs
	dec_input_concat = concatenate([encoder_out_combine, nwp_out], axis=-1)
	dec_input_concat = concatenate([dec_input_concat, times_out], axis=-1)


	dec_output, s_state, c_state = model.get_layer('lstm_1')(dec_input_concat, initial_state=[s_state_in, c_state_in])

	pred_test = model.get_layer('conv1d')(dec_output)
	pred_test = model.get_layer('conv1d_1')(pred_test)
	pred_test = model.get_layer('conv1d_2')(pred_test)

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

print(x1.shape)


# next_input = x1[0:1,:,:,:,0:1] 
# broadcaster = np.ones((1, output_seq_size, next_input.shape[2], next_input.shape[3], 1), dtype=np.float32)

for sample in range(30, 50):
	print(sample)

	# x1[sample:sample+1,:,:,:,0:1] = next_input 

	# spatial encoder
	ccn_enc_output = enoder_spatial_model.predict(x1[sample:sample+1,:,:,:,:])

	# lstm encoder
	encoder_out, enc_states = enoder_temporal_model.predict([x1[sample:sample+1,:,:,:,:], x2[sample:sample+1,:,:]])

	# if sample == 30:
	s_state, c_state = enc_states[0], enc_states[1]


	# call decoder
	# [x1[sample:sample+1,:,:,:,:], x2[sample:sample+1,:,:], x3[sample:sample+1,:,:], x4[sample:sample+1,:,:], s0[sample:sample+1,:], c0[sample:sample+1,:]]
	prediction, s_state, c_state = decoder_model.predict([ccn_enc_output, encoder_out, x3[sample:sample+1,:,:], x4[sample:sample+1,:,:],  s_state, c_state])
	

	if sample == 30:
		predictions = prediction
	else:
		predictions = np.concatenate([predictions, prediction], axis=0)


	# prediction_transform =  broadcaster * np.expand_dims(np.expand_dims(prediction, axis=-1), axis=-1)
	# next_input = np.concatenate([next_input, prediction_transform], axis=1)[0:1, -input_seq_size:, :, :, 0:1]




print('predicting')
testLen = 200
test_results = [] 
test_results_09 = [] 






for test_idx in range(testLen):
	print(f'test_idx: {test_idx}')

	predictions = model.predict([x1[test_idx:test_idx+1,:,:,:,:], x2[test_idx:test_idx+1,:,:], x3[test_idx:test_idx+1,:,:], x4[test_idx:test_idx+1,:,:]])
	predictions2 = model2.predict([x1[test_idx:test_idx+1,:,:,:,:], x2[test_idx:test_idx+1,:,:], x3[test_idx:test_idx+1,:,:], x4[test_idx:test_idx+1,:,:]])
	test_results.append(predictions[0])
	test_results_09.append(predictions2[0])

test_results = np.squeeze(np.concatenate(test_results, axis = 0))
test_results_09 = np.squeeze(np.concatenate(test_results_09, axis = 0))

print(test_results)

# save the predictions
# with open("./Data/wind/predictions.pkl", "wb") as y_hat:
# 	dump(predictions, y_hat)


idx = 0
plt.plot(test_results[idx:idx+168], label="prediction5")
plt.plot(test_results_09[idx:idx+168], label="prediction9")
plt.plot(y[idx:idx+168,:,0].flatten(), label="actual")
plt.legend()
plt.show()





# idx = 0
# plt.plot(predictions[idx:idx+168,:].flatten(), label="prediction_0.5")
# # plt.plot(predictions1[idx:idx+7,:].flatten(), label="prediction_0.1")
# # plt.plot(predictions2[idx:idx+7,:].flatten(), label="prediction_0.9")
# plt.plot(y[30:30+168,:,0].flatten(), label="actual")
# plt.legend()
# plt.show()











