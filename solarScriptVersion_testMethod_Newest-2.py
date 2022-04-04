import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
import sys, os
import h5py 

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Conv2D, Softmax, Bidirectional, Dense, TimeDistributed, LSTM 
from tensorflow.keras.layers import Input, Activation, AveragePooling2D, Lambda, concatenate, Flatten, BatchNormalization, RepeatVector, Permute, Lambda, Dropout, Average, Dot
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Reshape
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from pickle import dump, load
import time

np.set_printoptions(threshold=sys.maxsize)
tf.random.set_seed(180)


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import seaborn as sns
import random

###########################################_____LOAD_PROCESSED_DATA_____############################################
model_type ="price"

if model_type == 'wind':
	dataset_name = 'train_set_V100_wind.hdf5'
elif model_type == 'demand':
	dataset_name = 'dataset_V2_withtimefeatures_Demand.hdf5'
elif model_type == 'solar':
	dataset_name = 'dataset_solar_v30.hdf5'
elif model_type == 'price':
	dataset_name = 'dataset_V2_DAprice.hdf5'

# quantiles = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

quantiles = [0.01, 0.5, 0.99]

# load training data dictionary
# train_set_load = open("/content/drive/My Drive/Processed_Data/train_set_V6_withtimefeatures_120hrinput_.pkl", "rb") 
# train_set = load(train_set_load)
# train_set_load.close()

# print train_set data shapes
# for key in train_set.keys():
# 	print(f'{key}: {train_set[key].shape}')

# get size parameters
f = h5py.File(f"/content/drive/My Drive/Processed_Data/{dataset_name}", "r")
features = np.empty_like(f['train_set']['X1_train'][0])
times_in = np.empty_like(f['train_set']['X2_train'][0])
times_out = np.empty_like(f['train_set']['X3_train'][0])
labels = np.empty_like(f['train_set']['y_train'][0])
x_len = f['train_set']['X1_train'].shape[0]
y_len = f['train_set']['y_train'].shape[0]
print('size parameters loaded')

input_seq_size = 336
output_seq_size = 48

print(x_len)
print(y_len)


###########################################_____DATA_GENERATOR_____#################################################

params = {'batch_size': 16,
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
		(X_train1, X_train2, X_train3, X_train4, s0, c0), y_train = self.__data_generation(input_indexes, output_indexes)  

		y_trues = [y_train for i in quantiles]    

		y_trues.extend([[]]) 

		# print(len(y_trues))
		# print(y_trues[-1])        
		# sys.exit()        

		return (X_train1, X_train2, X_train3, X_train4, s0, c0), (y_trues) # pass empty training outputs to extract extract attentions

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

		f = h5py.File(f"/content/drive/My Drive/Processed_Data/{self.dataset_name}", "r")      
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

		s0 = np.zeros((self.batch_size, 32))
		c0 = np.zeros((self.batch_size, 32))

		# print(X_train1.shape)
		# print(y_train.shape)

     
		return (X_train1, X_train2, X_train3, X_train4, s0, c0), y_train

training_generator = DataGenerator(dataset_name = dataset_name, x_length = x_len, y_length = y_len,  **params)


###########################################_____MODEL_ARCHITECTURE_____#################################################

# cpature some usful dimensions
Tx = input_seq_size
Ty = output_seq_size

if model_type != "price":
	height, width, channels = features.shape[0], features.shape[1], features.shape[2]
else:
	channels = features.shape[-1]

times_in_dim = times_in.shape[-1]
times_out_dim = times_out.shape[-1]
n_s = 32


#one-step temporal Atttention
# class attention(tf.keras.layers.Layer):

# 	def __init__(self, hidden_units, **kwargs):
# 		# super(attention, self).__init__(hidden_units)
# 		self.hidden_units = hidden_units
# 		super(attention, self).__init__(**kwargs)


# 	def build(self, input_shape):

# 		self.conv1d_1 = Conv1D(self.hidden_units, kernel_size=1, strides=1, padding='same', activation='relu')
# 		self.conv1d_2 = Conv1D(self.hidden_units, kernel_size=1, strides=1, padding='same', activation='relu')
# 		self.conv1d_3 = Conv1D(1, kernel_size=1, strides=1, padding='same', activation='linear')

# 		self.tanh = tf.keras.layers.Activation("tanh")
# 		self.alphas = Softmax(axis = 1, name='attention_weights')
		
# 		super(attention, self).build(input_shape)

# 	def call(self, enc_output, h_state, c_state):

# 		h_state_time = K.expand_dims(h_state, axis=1)
# 		c_state_time = K.expand_dims(c_state, axis=1)
# 		hc_state_time = concatenate([h_state_time, c_state_time], axis=-1)

# 		x1 = self.conv1d_1(enc_output)
# 		x2 = self.conv1d_2(hc_state_time)
# 		x3 = self.tanh(x1 + x2)
# 		x4 = self.conv1d_3(x3)
# 		attn_w = self.alphas(x4) 

# 		context = attn_w * enc_output
# 		context = tf.reduce_sum(context, axis=1)
# 		context = K.expand_dims(context, axis=1)
# 		# context = RepeatVector(Ty)(context)

# 		return [attn_w, context]

# 	def compute_output_shape(self):
# 		return [(input_shape[0], Tx, 1), (input_shape[0], 1, n_s)]

# 	def get_config(self):
# 		config = super(attention, self).get_config()
# 		config.update({"hidden_units": self.hidden_units})
# 		return config

class attention(tf.keras.layers.Layer):

	def __init__(self, hidden_units, **kwargs):
		# super(attention, self).__init__(hidden_units)
		self.hidden_units = hidden_units
		super(attention, self).__init__(**kwargs)


	def build(self, input_shape):

		input_dim = int(input_shape[-1])

		self.attention_score_vec = Dense(64, name='attention_score_vec')
		self.h_t = Dense(64, name='ht')
		# self.h_t = Lambda(lambda x: x[:, -1, :], output_shape=(input_dim,), name='last_hidden_state')
		self.attention_score = Dot(axes=[1, 2], name='attention_score')
		self.attention_weight = Activation('softmax', name='attention_weight')
		self.context_vector = Dot(axes=[1, 1], name='context_vector')
		# self.attention_output = concatenate(name='attention_output')
		self.attention_vector = Dense(self.hidden_units, activation='tanh', name='attention_vector')

		super(attention, self).build(input_shape)

	def call(self, enc_output, enc_out, h_state, c_state):


		score_first_part = self.attention_score_vec(enc_output)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
		# h_t = self.h_t(enc_output)
		h_t = concatenate([h_state, enc_out[:,0,:]])
		h_t = self.h_t(h_t)

		score = self.attention_score([h_t, score_first_part])

		attention_weights = self.attention_weight(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
		context_vector = self.context_vector([enc_output, attention_weights])
		pre_activation = concatenate([context_vector, h_t])
		attention_vector = self.attention_vector(pre_activation)

		attention_weights = K.expand_dims(attention_weights, axis=-1)
		attention_vector = K.expand_dims(attention_vector, axis=1)

		return [attention_weights, attention_vector]

	def compute_output_shape(self):
		return [(input_shape[0], Tx, 1), (input_shape[0], 1, n_s)]

	def get_config(self):
		config = super(attention, self).get_config()
		config.update({"hidden_units": self.hidden_units})
		return config









# instantiate attention layers
# temporal_attn = attention(n_s, name="temporal_attention")
# spatial_attn = attention(n_s, name="spatial_attention")


def cnn_encoder(ccn_input):
    # input shape -> (batch, time, width, height, features)
    # output shape -> (batch, time, width x height, embedding_size)

	ccn_enc_output = TimeDistributed(Conv2D(16, kernel_size=3, strides=1, activation="relu"))(ccn_input)
	ccn_enc_output = TimeDistributed(AveragePooling2D(pool_size=(2, 2), data_format="channels_last"))(ccn_enc_output)
	# ccn_enc_output = BatchNormalization()(ccn_enc_output)    
	ccn_enc_output = TimeDistributed(Conv2D(32, kernel_size=3, strides=1, activation="relu"))(ccn_enc_output)
	ccn_enc_output = TimeDistributed(Conv2D(64, kernel_size=3, strides=1, activation="relu"))(ccn_enc_output)
	# ccn_enc_output = BatchNormalization()(ccn_enc_output)  
	ccn_enc_output = TimeDistributed(Conv2D(128, kernel_size=3, strides=1, activation="relu"))(ccn_enc_output)

	ccn_enc_output = Reshape((ccn_enc_output.shape[1], -1, ccn_enc_output.shape[-1]))(ccn_enc_output) 

	ccn_enc_output = K.mean(ccn_enc_output, axis=1) 

	# print(ccn_enc_output.shape)

    
	return ccn_enc_output

# encoder layers
lstm_encoder = Bidirectional(LSTM(n_s*2, return_sequences = True, return_state = True))
# lstm_encoder = LSTM(n_s, return_sequences = True, return_state = True)

def encoder(input, times_in):
    
	if model_type != "price":
		enc_output = K.mean(input, axis=(2,3))
	else:
		enc_output = input       

	# concat input time features with input
	enc_output = concatenate([enc_output, times_in], axis=-1)

	enc_output, forward_h, forward_c, backward_h, backward_c = lstm_encoder(enc_output)
	# enc_output, enc_h, enc_s = lstm_encoder(enc_output)

	enc_h = concatenate([forward_h, backward_h], axis=-1)
	enc_s = concatenate([forward_c, backward_c], axis=-1)

	# # concat input time features with input
	# enc_output = concatenate([enc_output, times_in], axis=-1)

	return enc_output, enc_h, enc_s

lstm_decoder = LSTM(n_s, return_sequences = True, return_state = True)

def decoder(context, h_state, cell_state):

    # concat encoder input and time features
	# context = concatenate([context, times_out], axis=-1)
    
	dec_output, h_state , c_state = state = lstm_decoder(context, initial_state = [h_state, cell_state])
# decoder = LSTM(n_s, return_sequences = True, return_state = True)

	return dec_output, h_state, c_state

# make custom activation - swish
from keras.backend import sigmoid

def swish(x, beta = 1):
	return (x * sigmoid(beta * x))



# Getting the Custom object and updating them
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
  
# Below in place of swish you can take any custom key for the name 
get_custom_objects().update({'swish': Activation(swish)})
  

# layer for output predictions
# predict_1 = Conv1D(64, kernel_size=1, strides=1, padding="same", activation="swish", name='conv1_q')
# predict_2 = Conv1D(16, kernel_size=1, strides=1, padding="same", activation="swish", name='conv2_q')
# predict_3 = Conv1D(1, kernel_size=1, strides=1, padding="same", activation="linear", name='conv3_q')

# predict_1 = Dense(64, activation="swish")
# predict_2 = Dense(16,  activation="swish")
# predict_3 = Dense(1, activation="swish")



# dropout
drop = Dropout(0.3)

# predict_1 = Dense(32, activation="linear")
# predict_2 = Dense(1, activation="linear")

# full model


# define inputs
if model_type != "price":
	x_input = Input(shape=(Tx, height, width, channels))
else:
	x_input = Input(shape=(Tx, channels))

times_in = Input(shape=(Tx, times_in_dim))
times_out = Input(shape=(Ty, times_out_dim))
out_nwp = Input(shape=(Ty, channels-1))
s_state0 = Input(shape=(32,))
c_state0 = Input(shape=(32,))
# dec_inp = Input(shape=(None, 1))

s_state = s_state0
c_state = c_state0

# create empty list for outputs
all_predictions = []
all_temp_attn_weights = []
all_spat_attn_weights = []


qunatile_predictions = []

temporal_attns = [] 
spatial_attns = [] 

decoders_quantiles = {}
temporal_attn_quantiles = {}


# for q in quantiles:

# call CCN_encoder function
if model_type != "price":
	ccn_enc_output = cnn_encoder(x_input)

# call LSTM_encoder function 
lstm_enc_output, enc_s_state, enc_c_state = encoder(x_input, times_in)

# s_state_enc = enc_s_state
# c_state_enc = enc_c_state

# y_prev = K.mean(x_input, axis=(2,3))
# y_prev = y_prev[:, -49, 0:1]
# define intial decoder input
# y_prev = lstm_enc_output[:,-1,-1]
# y_prev = K.expand_dims(y_prev, axis=1)
# y_prev = K.expand_dims(y_prev, axis=-1)
# y_prev = K.expand_dims(y_prev, axis=-1)

# for t in range(Ty):

# get context matrix (temporal)
# attn_weights_temp, context_temp = temporal_attn(lstm_enc_output, s_state, c_state)

# get context matrix (spatial)
# attn_weights_spat, context_spat = spatial_attn(ccn_enc_output, s_state, c_state)

# combine spatial and temporal context
# context = concatenate([context_temp, context_spat], axis=-1) 

# print(context.shape)

# decoder_input = concatenate([context, times_out], axis=-1) 
# decoder_input = concatenate([decoder_input, out_nwp], axis=-1)  

# get the previous value for teacher forcing
# decoder_input = concatenate([context, y_prev], axis=-1)

# decoder_input = drop(decoder_input)

# output times for input to decoder
# times_out_single = times_out[:,t,:]
# times_out_single = K.expand_dims(times_out_single , axis=1)

# decoder_input = Dropout(0.2)(decoder_input)
# decoder_input = concatenate([decoder_input, times_out_single], axis=-1)


	# qunatile_predictions = []

	# temporal_attns = [] 
	# spatial_attns = [] 

	# decoders_quantiles = {}
	# temporal_attn_quantiles = {}


# prediction = lstm_enc_output[:,-1:,:]

# call decoder
for q in quantiles: 
  
	ts_predictions = []
	temp_attns = []
	spatial_attns = []

	if model_type != "price":
		intial_in = K.mean(x_input, axis=(2,3))
		prev_prediction = intial_in[:,-1:,0:1]

	decoder = LSTM(32, return_sequences = False, return_state = True, name=f'decoder_q_{q}')
	# decoder_2 = LSTM(32, return_sequences = False, return_state = True, name=f'decoder2_q_{q}')
	spatial_attention = attention(n_s, name=f"spatial_attention_q_{q}")
	temporal_attention = attention(n_s, name=f"temporal_attention_q_{q}")

	output_1 = Dense(32, activation="swish", name=f'dense1_q_{q}')
	output_2 = Dense(1, name=f'dense3_q_{q}')
	final_act = Activation('relu', name=f'swish_act_q_{q}')

    # concatenation layers

	s_state = s_state0
	c_state = c_state0

	for ts in range(Ty):

		# current_nwp_data = Lambda(lambda x: x[:,ts:ts+1,:], name=f"current_nwp_q_{q}_{ts}")(out_nwp)
		# current_times_out = Lambda(lambda x: x[:,ts:ts+1,:], name=f"current_times_out_q_{q}_{ts}")(times_out)

		enc_out = concatenate([out_nwp[:,ts:ts+1,:], times_out[:,ts:ts+1,:]], axis=-1, name=f'concat1_q_{q}_{ts}')        

		# get context matrix (temporal)
		# attn_weights_temp, context_temp = attention(n_s, name=f"temporal_attention_q_{q}_{ts}")(lstm_enc_output, current_times_out, s_state, c_state)
		attn_weights_temp, context = temporal_attention(lstm_enc_output, enc_out, s_state, c_state)

		# get context matrix (spatial)
		# attn_weights_spat, context_spat = attention(n_s, name=f"spatial_attention_q_{q}_{ts}")(ccn_enc_output, s_state, c_state)
		if model_type != "price":
			attn_weights_spat, context_spat = spatial_attention(ccn_enc_output, enc_out, s_state, c_state)

			# combine spatial and temporal context
			context = concatenate([context, context_spat], axis=-1, name=f'concat1.5_q_{q}_{ts}') 
			# context = context_temp

			if ts > 0:
				decoder_input = concatenate([out_nwp[:,ts-1:ts,:], times_out[:,ts-1:ts,:]], axis=-1, name=f'concat2_q_{q}_{ts}')
			else:
				decoder_input = concatenate([intial_in[:,-1:,1:], times_in[:,-1:,:]], axis=-1, name=f'concat3_q_{q}_{ts}')  
		else:
			decoder_input = times_out[:,ts-1:ts,:]                              

		# decoder_input = concatenate([prev_prediction, decoder_input], axis=-1, name=f'concat4_q_{q}_{ts}')
		# decoder_input = concatenate([decoder_input, context], axis=-1, name=f"concat2_q_{q}_{ts}") 

		# if ts > 0:
		
        # decoder_input = concatenate([decoder_input, K.expand_dims(prev_prediction, axis=1)], axis=-1, name=f"concat3_q_{q}_{ts}")  

		# print(decoder_input.shape)

		dec_output, s_state, c_state = decoder(decoder_input, initial_state = [s_state, c_state])
		# dec_output, s_state, c_state = decoder(decoder_input)
		# dec_output, s_state, c_state = state = LSTM(n_s * 2, return_sequences = False, return_state = True, name=f'decoder_q_{q}_{ts}')(decoder_input, initial_state = [s_state, c_state])
		# dec_output, _ , _ = state = LSTM(n_s, return_sequences = True, return_state = True, name=f'decoder_q_{q}')(decoder_input)

		# predict_2 = Conv1D(16, kernel_size=1, strides=1, padding="same", activation="swish", name=f'conv2_q_{q}')(predict_1)
		prediction = concatenate([context, K.expand_dims(dec_output,axis=1)], axis=-1, name=f'concat5_q_{q}_{ts}')

		# output = Dense(32, activation="linear", name=f'dense1_q_{q}_{ts}')(prediction)
		output = output_1(prediction)
		# output = Dropout(0.5)(output)

		# output = Dense(1, name=f'dense3_q_{q}_{ts}')(output)
		output = output_2(output)

		# output = Activation('relu', name=f'swish_act_q_{q}_{ts}')(output)
		output = final_act(output)

		prev_prediction = output
		ts_predictions.append(output)
		temp_attns.append(attn_weights_temp)

		if model_type != "price":        
			spatial_attns.append(attn_weights_spat)

	ts_predictions_total = concatenate(ts_predictions, axis = 1)
	temp_attns_total = concatenate(temp_attns, axis = -1)

	if model_type != "price":
		sptial_attns_total = concatenate(spatial_attns, axis = -1)

	qunatile_predictions.append(ts_predictions_total)


qunatile_predictions.extend([temp_attns_total])

if model_type != "price":
	qunatile_predictions.extend([sptial_attns_total])

# sys.exit()

# predictions = K.expand_dims(predictions , axis=-1)

	# output = Conv1D(64, kernel_size=1, strides=1, padding="same", activation="relu", name=f"1conv_{t}")(dec_output)
	# output = Conv1D(16, kernel_size=1, strides=1, padding="same", activation="relu", name=f"2conv_{t}")(output)
	# output = Conv1D(1, kernel_size=1, strides=1, padding="same", activation="linear", name=f"3conv_{t}")(output)

	# if t == 0:
	# 	predictions = output
	# 	attentions_temp = attn_weights_temp
	# 	attentions_spat = attn_weights_spat
	# else:
	# 	predictions = concatenate([predictions, output], axis=1, name='predictions_{}'.format(t))
	# 	attentions_temp = concatenate([attentions_temp, attn_weights_temp], axis=-1, name='attentions_temp_{}'.format(t))
	# 	attentions_spat = concatenate([attentions_spat, attn_weights_spat], axis=-1, name='attentions_spat_{}'.format(t))


	# get previous actual value for teacher forcing
	# if random.random() < 0.5:
	# 	print('real')       
	# y_prev = dec_inp[:, t, :]
	# y_prev = K.expand_dims(y_prev , axis=1) 
	# else:
	# 	print('prediction') 
	# y_prev = output
	# y_prev = Reshape((1, 1))(y_prev) 
	# y_prev = output
	# y_prev = drop(y_prev)   
	# y_prev = Reshape((1, 1))(y_prev) 
	# print(f'y_prev: {y_prev.shape}')

	# y_prev = tf.identity(output)
	# y_prev = Reshape((1, 1))(y_prev) 

	# append output data
	# all_predictions.append(output)
	# all_temp_attn_weights.append(attn_weights_temp)
	# all_spat_attn_weights.append(attn_weights_spat)


# all_predictions = tf.concat([i for i in all_predictions], axis=0)
# all_temp_attn_weights = tf.concat([i for i in all_temp_attn_weights], axis=0)
# all_spat_attn_weights = tf.concat([i for i in all_spat_attn_weights], axis=0)



# predictions = Lambda(lambda x: concatenate(x, axis=1))(all_predictions)
# attention_temporal = Lambda(lambda x: concatenate(x, axis=1))(all_temp_attn_weights)
# attention_spatial = Lambda(lambda x: concatenate(x, axis=1))(all_spat_attn_weights)

model = Model(inputs = [x_input, times_in, times_out, out_nwp, s_state0, c_state0], outputs = qunatile_predictions)


# define the pinball loss function to optimise
def defined_loss(q,y,f):
	e = (y - f)
	return K.mean(K.maximum(q*e, (q-1) * e), axis = -1)


# declare quantiles
# quantiles = ['0.' + str(i) for i in range(1,10)]
# quantiles = list(map(float, quantiles))
# quantiles.append(0.99)
# quantiles.insert(0, 0.01)
# quantiles = [0.5, 0.1, 0.9]
# print(quantiles)

#include clipvalue in optmisier
optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.001)

# lstm_idx = [x for x in range(1,49)]
# checking final architecture
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot


# model_to_dot(model).create(prog='dot', format='svg')
# plt.show()
# tf.keras.utils.plot_model(model, to_file='dot_img_file.png', show_shapes=True)
# sys.exit()


# function to set only decoder to trainable
def freeze_decoder_train(model):
    for layer in model.layers:
        # print(f'*****************************: {layer.name[:4]}')
        if (layer.name[:5] == 'lstm_') or (layer.name == 'conv1d'):
            print('ACTIVATED')
            layer.trainable = True 
        else:
            layer.trainable = False 
    return model


# os.mkdir('/content/drive/My Drive/solar_models')



# for q in quantiles:
# 	# make unique folder for each folder
# 	os.mkdir(f'/content/drive/My Drive/quantile_{q}')
# 	# check if central estimate model exists
# 	if os.path.isfile('/content/drive/My Drive/quantile_0.5/solarGeneration_forecast_MainModel_Q_0.5.h5'):
# 		print('0.5 model present')   
# 		model = load_model('/content/drive/My Drive/quantile_0.5/solarGeneration_forecast_MainModel_Q_0.5.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention})
# 		# re-compile model with new loss for current quantile 
# 		model.compile(loss = [lambda y,f: defined_loss(q,y,f), None, None], optimizer= optimizer, metrics = ['mae'])
# 		# load trained weights from q_0.5
# 		model.load_weights(f'/content/drive/My Drive/quantile_0.5/solarGeneration_forecast_weights_Q_0.5.h5')
# 		# freeze all weights apart from decoder fro re-train
# 		model = freeze_decoder_train(model)
# 		print(model.summary())
# 		model.fit(training_generator, epochs = 10)
# 		enc_model_temp_test.save(f'/content/drive/My Drive/quantile_{q}/solarGeneration_encoderModelTemporal_Q_{q}.h5')
# 		enc_model_spat_test.save(f'/content/drive/My Drive//quantile_{q}/solarGeneration_encoderModelSpatial_Q_{q}.h5')
# 		# model.save(f'/content/drive/My Drive/quantile_{q}/solarGeneration_forecast_MainModel_Q_{q}.h5')
# 	else:
# 		print('New Model') 
# 		# train median model for reference to other qunatiles
# 		model.compile(loss = [lambda y,f: defined_loss(q,y,f), None, None], optimizer= optimizer, metrics = ['mae'])
# 		print(model.summary())
# 		# sys.exit()
# 		model.fit(training_generator, epochs = 10)
# 		# save wegihts 
# 		model.save_weights(f'/content/drive/My Drive/quantile_{q}/solarGeneration_forecast_weights_Q_{q}.h5')
# 		# save some additional models for inference
# 		enc_model_temp_test = Model(inputs = [x_input, times_in], outputs=[lstm_enc_output])
# 		enc_model_spat_test = Model(x_input, ccn_enc_output)
# 		enc_model_temp_test.save(f'/content/drive/My Drive/quantile_{q}/solarGeneration_encoderModelTemporal_Q_{q}.h5')
# 		enc_model_spat_test.save(f'/content/drive/My Drive//quantile_{q}/solarGeneration_encoderModelSpatial_Q_{q}.h5')
# 		# model.save(f'/content/drive/My Drive/quantile_{q}/solarGeneration_forecast_MainModel_Q_{q}.h5')

# 	model.save(f'/content/drive/My Drive/quantile_{q}/solarGeneration_forecast_MainModel_Q_{q}.h5')	
# 	K.clear_session()
		
def QuantileLoss(perc, delta=1e-4):
    perc = np.array(perc).reshape(-1)
    perc.sort()
    perc = perc.reshape(1, -1)
    def _qloss(y, pred):
        I = tf.cast(y <= pred, tf.float32)
        error = K.abs(y - pred)
        correction = I * (1 - perc) + (1 - I) * perc
        # huber loss
        huber_loss = K.sum(correction * tf.where(error <= delta, 0.5 * error ** 2 / delta, error - 0.5 * delta), -1)
        print(huber_loss.shape)
        # order loss
        q_order_loss = K.sum(K.maximum(0.0, pred[:,:, :-1] - pred[:,:, 1:] + 1e-6), -1)
        print(q_order_loss.shape)
        return huber_loss + q_order_loss
    return _qloss


perc_points = [0.01, 0.25, 0.5, 0.75, 0.99]

def loss_func(q):
    return lambda y,f: defined_loss(q,y,f) 


func_test = lambda y,f: defined_loss(q,y,f) 

a = lambda y,f: defined_loss(0.01,y,f)
b = lambda y,f: defined_loss(0.5,y,f) 
c = lambda y,f: defined_loss(0.99,y,f)

q_losses = [a, b, c]

q_losses.append(None)
print(q_losses)
print(func_test)

# sys.exit()
q = 'all'
# train each model for each quantile
# for q in quantiles:
# 	print(q)
	# model = solarGenation_Model()
model.compile(loss = q_losses, optimizer= optimizer)
# model.compile(loss = [lambda y,f: defined_loss(q,y,f), None, None], optimizer= optimizer, metrics = ['mae'])
print(model.summary())

train = model.fit(training_generator, epochs = 10)

os.mkdir(f'/content/drive/My Drive/{model_type}Models/q_{q}')
# model_freeze.save_weights('/content/drive/My Drive/solarGeneration_forecast_weights_freezed' + '_Q_%s' %(q) + '.h5')
model.save(f'/content/drive/My Drive/{model_type}Models/q_{q}/{model_type}Generation_forecast_MainModel_Q_{q}.h5')
# model.save_weights('/content/drive/My Drive/solarGeneration_forecast_weights_test2' + '_Q_%s' %(q) + '.h5')

# save some additional models for inference
enoder_temporal_model = Model(inputs = [x_input, times_in], outputs=[lstm_enc_output, enc_s_state, enc_c_state])
enoder_spatial_model = Model(x_input, ccn_enc_output)
enoder_temporal_model.save(f'/content/drive/My Drive/{model_type}Models/q_{q}/{model_type}Generation_encoderModelTemporal' + '_Q_%s' %(q) + '.h5')
enoder_spatial_model.save(f'/content/drive/My Drive/{model_type}Models/q_{q}/{model_type}Generation_encoderModelSpatial' + '_Q_%s' %(q) + '.h5')
print('predicting')
predictions = model.predict(training_generator)
# predictions = predictions[0]

# with open(f"/content/drive/My Drive//windModels/predictions_{q}.pkl", "wb") as y_hat:
#     dump(predictions, y_hat)
# K.clear_session()


print('predicting')
predictions = model.predict(training_generator)
predictions1 = predictions[0]
predictions2 = predictions[1]
predictions3 = predictions[2]

idx = 84
plt.plot(predictions1[idx:idx+7,:].flatten(), label="0.1")
plt.plot(predictions2[idx:idx+7,:].flatten(), label="0.5")
plt.plot(predictions3[idx:idx+7,:].flatten(), label="0.9")
# plt.plot(y[idx:idx+7,:,0].flatten(), label="actual")
plt.legend()
plt.show()


sys.exit()

# empty dictionaries for decoder models
decoder_models, enoder_temporal_models, enoder_spatial_models = {}, {}, {}





######## model for inference #############
def inference_model(idx):

	# LSTM Encoder
	enc_model_temp_test = Model(inputs = [x_input, times_in], outputs=[lstm_enc_output])
	# CNN Encoder
	enc_model_spat_test = Model(x_input, ccn_enc_output)

	# Encoder outputs for setup
	ccn_enc_output_test = Input(shape=(320, 256))
	lstm_enc_output_test = Input(shape=(Tx, n_s + times_in_dim))

	# Decoder Input
	dec_input_test = Input(shape=(1, 1))
	dec_input_test_int = Input(shape=(1, n_s + times_in_dim))
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
	dec_output, s_state, c_state = model.get_layer('lstm_1')(dec_input_concat_test, initial_state=[s_state0, c_state0])
	# if idx == 1:
	# 	dec_output, s_state, c_state = model.get_layer('lstm_1')(dec_input_concat_test_int, initial_state=[s_state0, c_state0])
	# else:   
	# 	dec_output, s_state, c_state = model.get_layer(f'lstm_{idx}')(dec_input_concat_test, initial_state=[s_state0, c_state0])

	pred_test = predict_1(dec_output)
	pred_test = predict_2(pred_test)

	# Inference Model
	deoceder_test_model = Model(inputs=[dec_input_test, dec_input_test_int, times_out_test, lstm_enc_output_test, ccn_enc_output_test, s_state0, c_state0], outputs=[pred_test, s_state, c_state, attn_weights_temp_test, attn_weights_spat_test])  
	     
	return deoceder_test_model


# for idx in range(1, Ty+1):
# 	enoder_temporal_model, enoder_spatial_model, decoder_models[f'{idx}'] = inference_model(idx)
decoder_inference = inference_model(1)
# print(decoder_models)
# print(decoder_models[f'{idx}'][0])
# sys.exit()

print('decoder loaded')
############_________Inference__________##############
sample_size = 200
# main_input = train_set['X1_train'][:sample_size,:,:,:,:]
# times_in = train_set['X2_train'][:sample_size,:,:]
# times_out = train_set['X3_train'][:sample_size,:,:]
# y_train = train_set['y_train'][:sample_size,:,:]
s0 = np.zeros((sample_size, n_s))
c0 = np.zeros((sample_size, n_s))

s_state = s0
c_state = c0
predictions = []

# inference run
enc_temp_out = enoder_temporal_model.predict([train_set['X1_train'][:sample_size,:,:,:,:], train_set['X2_train'][:sample_size,:,:]])
enc_spat_out = enoder_spatial_model.predict(train_set['X1_train'][:sample_size,:,:,:,:])

print('encoder models ran')

# intial decoder input
dec_input_int = enc_temp_out[:,-1,:]
dec_input_int = K.expand_dims(dec_input_int, axis=1)
# dec_input = np.zeros((sample_size, 1, 1))

dec_input = enc_temp_out[:,-1,-1]
dec_input = K.expand_dims(dec_input, axis=1)
dec_input = K.expand_dims(dec_input, axis=-1)

for t in range(1, Ty+1):

	print(t)

	# get current 'times out' reference
	times_out_single = times_out[:,t-1,:]
	times_out_single = K.expand_dims(times_out_single, axis=1)

	# print(dec_input.shape)
	# print(times_out_single.shape)
	# print(enc_temp_out.shape)
	# print(enc_spat_out.shape)
	# print(s_state.shape)
	# print(c_state.shape)

    # dec_output, s_state, c_state = decoder(decoder_input, times_out_single, s_state, c_state)
	prediction, s_state, c_state, temporal_attention, spatial_attention = decoder_inference.predict([dec_input, dec_input_int, times_out_single, enc_temp_out, enc_spat_out, s_state, c_state])

	# dec_input = y_train[:,t-1,:]
	dec_input = prediction
	# print('dec_input')
	# print(dec_input.shape)

	# predictions.append(prediction)
    
	if t == 1:
		predictions_test = prediction
		total_temporal_attn_test = temporal_attention
		total_spatial_attn_test = spatial_attention
	else:
		predictions_test = np.concatenate([predictions_test, prediction], axis=1)
		total_temporal_attn_test = np.concatenate([total_temporal_attn_test, temporal_attention], axis=-1)
		total_spatial_attn_test = np.concatenate([total_spatial_attn_test, spatial_attention], axis=-1)
	

	print(total_temporal_attn_test.shape)
	print(total_spatial_attn_test.shape)


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



# input1 = train_set['X1_train'][:sample_size,:,:,:,:]
# input2 = train_set['X2_train'][:sample_size,:,:]
# input3 = train_set['X3_train'][:sample_size,:,:]
# outputs = train_set['y_train'][:sample_size,:,:]

# n_s = 128

s0 = np.zeros((sample_size, n_s))
c0 = np.zeros((sample_size, n_s))


# performance against training data
inputs = [train_set['X1_train'][:sample_size,:,:,:,:], train_set['X2_train'][:sample_size,:,:], train_set['X3_train'][:sample_size,:,:], s0, c0, train_set['y_train'][:sample_size,:,:]]

print('predicitng...')
yhat1 = model.predict(inputs)
model_out1 = yhat1[0]
attentions_temp_train = yhat1[1]


idx = 884
x = np.average(train_set['X1_train'], axis=(2,3))[idx, :]
y = outputs[idx, :, 0]
y_hat_train = model_out1[idx]
y_hat_test = predictions_test[idx]

print('prediction_test_Size')
print(predictions_test.shape)


print(x.shape)
print(y.shape)
print(y_hat_train.shape)
print(y_hat_test.shape)



# train temporal attention
att_w_temp_train = np.transpose(attentions_temp_train[idx])
# test temporal attention
att_w_temp_test = np.transpose(total_temporal_attn_test[idx])


temporal_attention_graph(x, y, y_hat_train, att_w_temp_train)

temporal_attention_graph(x, y, y_hat_test, att_w_temp_test)





















