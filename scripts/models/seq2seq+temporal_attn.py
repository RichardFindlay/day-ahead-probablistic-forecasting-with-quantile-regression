import numpy as np
import sys, os
import h5py 
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Bidirectional, Dense, TimeDistributed, LSTM 
from tensorflow.keras.layers import Input, Activation, AveragePooling2D, Lambda, concatenate, Reshape
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects

# import custom classes 
from _shared.attention_layer import attention
from _shared.timeseries_data_generator import DataGenerator

np.set_printoptions(threshold=sys.maxsize)
tf.random.set_seed(180)

###########################################_____SET_MODEL_PARAMETERS_____############################################
model_type ="solar"

# declare dataset file
dataset_name = f'dataset_{model_type}.hdf5'

# declare quantiles for model
quantiles = [0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95]

# get useful size parameters
f = h5py.File(f"../../data/processed/{model_type}/{dataset_name}", "r")
features = np.empty_like(f['train_set']['X1_train'][0])
times_in = np.empty_like(f['train_set']['X2_train'][0])
times_out = np.empty_like(f['train_set']['X3_train'][0])
labels = np.empty_like(f['train_set']['y_train'][0])
x_len = f['train_set']['X1_train'].shape[0]
y_len = f['train_set']['y_train'].shape[0]
f.close()

# input / output sequence sizes
input_seq_size = 336
output_seq_size = 48
n_s = 32 # number of hidden states used through model

###########################################_____DATA_GENERATOR_____#################################################

# data generator input parameters - avoid shuffle in this case 
params = {'batch_size': 16,
		'shuffle': False } 

# instantiate data generator object
training_generator = DataGenerator(dataset_name = dataset_name, x_length = x_len, y_length = y_len, hidden_states = n_s,  **params)

###########################################_____MODEL_ARCHITECTURE_____#################################################

# cpature some more useful dimensions
Tx = input_seq_size
Ty = output_seq_size

if model_type != "price":
	height, width, channels = features.shape[0], features.shape[1], features.shape[2]
else:
	channels = features.shape[-1]

times_in_dim = times_in.shape[-1]
times_out_dim = times_out.shape[-1]


# temporal encoder layers
lstm_encoder = Bidirectional(LSTM(n_s*2, return_sequences = True, return_state = True))

def encoder(input, times_in):
    
    # accomodate for case without 2D dataset
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

	# concat input time features with input
	# enc_output = concatenate([enc_output, times_in], axis=-1)

	return enc_output, enc_h, enc_s

# declare decoder layer
lstm_decoder = LSTM(n_s, return_sequences = True, return_state = True)

def decoder(context, h_state, cell_state):

    # concat encoder input and time features
	# context = concatenate([context, times_out], axis=-1)
    
	dec_output, h_state , c_state = state = lstm_decoder(context, initial_state = [h_state, cell_state])

	return dec_output, h_state, c_state

# make custom activation - swish
def swish(x, beta = 1):
	return (x * sigmoid(beta * x))

# add swish activation to keras
get_custom_objects().update({'swish': Activation(swish)})
  
# define inputs for model
x_input = Input(shape=(Tx, channels))

times_in = Input(shape=(Tx, times_in_dim))
times_out = Input(shape=(Ty, times_out_dim))
out_nwp = Input(shape=(Ty, channels-1))
s_state0 = Input(shape=(32,))
c_state0 = Input(shape=(32,))

# create empty list for outputs
qunatile_predictions = []
temporal_attns = [] 

# call LSTM_encoder function 
lstm_enc_output, enc_s_state, enc_c_state = encoder(x_input, times_in)

# call decoder
for q in quantiles: 
  	
  	# reset model parameters for each qunatile prediction
	ts_predictions = []
	temp_attns = []
	spatial_attns = []

	if model_type != "price":
		intial_in = K.mean(x_input, axis=(2,3))
		prev_prediction = intial_in[:,-1:,0:1]

	decoder = LSTM(32, return_sequences = False, return_state = True, name=f'decoder_q_{q}')
	spatial_attention = attention(n_s, name=f"spatial_attention_q_{q}")
	temporal_attention = attention(n_s, name=f"temporal_attention_q_{q}")

	output_1 = Dense(32, activation="swish", name=f'dense1_q_{q}')
	output_2 = Dense(1, name=f'dense3_q_{q}')
	final_act = Activation('relu', name=f'relu_act_q_{q}')

    # reset hidden states
	s_state = s_state0
	c_state = c_state0

	# make prediction for each output timestep
	for ts in range(Ty):

		if model_type != "price":
			enc_out = concatenate([out_nwp[:,ts:ts+1,:], times_out[:,ts:ts+1,:]], axis=-1, name=f'concat1_q_{q}_{ts}')        
		else:
			enc_out = times_out[:,ts:ts+1,:] 

		# get context matrix (temporal)
		attn_weights_temp, context = temporal_attention(lstm_enc_output, enc_out, s_state, c_state)

		# get context matrix (spatial)
		if model_type != "price":

			# make decoder input - nwp + time features if not price predictions, other wise just time features 
			if ts > 0:
				decoder_input = concatenate([out_nwp[:,ts-1:ts,:], times_out[:,ts-1:ts,:]], axis=-1, name=f'concat2_q_{q}_{ts}')
			else:
				decoder_input = concatenate([intial_in[:,-1:,1:], times_in[:,-1:,:]], axis=-1, name=f'concat3_q_{q}_{ts}')  
		else:
			if ts > 0:
				decoder_input = times_out[:,ts-1:ts,:]
			else:
				decoder_input = times_in[:,-1:,:]                             

		# call decoder 
		dec_output, s_state, c_state = decoder(decoder_input, initial_state = [s_state, c_state])

		# combine context with decoder output
		prediction = concatenate([context, K.expand_dims(dec_output,axis=1)], axis=-1, name=f'concat5_q_{q}_{ts}')

		# pass through MLP
		output = output_1(prediction)
		output = output_2(output)

		if model_type == "solar":
			output = final_act(output)

		# collect outputs for final predictions
		prev_prediction = output
		ts_predictions.append(output)
		temp_attns.append(attn_weights_temp)

	ts_predictions_total = concatenate(ts_predictions, axis = 1)
	temp_attns_total = concatenate(temp_attns, axis = -1)

	qunatile_predictions.append(ts_predictions_total)

# append spatial and temporal predictions - if using final model as inference
# qunatile_predictions.extend([temp_attns_total])
# qunatile_predictions.extend([sptial_attns_total])

# instantiate model
model = Model(inputs = [x_input, times_in, times_out, out_nwp, s_state0, c_state0], outputs = qunatile_predictions)


###########################################_____MODEL_TRAINING_____#################################################

#include clipvalue in optmisier
optimizer = tensorflow.keras.optimizers.Adam(learning_rate = 0.001)

# define loss for each quantile
q_losses = [lambda y,f: K.mean(K.maximum(q*(y - f), (q-1) * (y - f)), axis = -1) for q in quantiles]

# append additional empty losses for temporal and spatial encoders
# q_losses.append([None,None])

# compile and train model
model.compile(loss = q_losses, optimizer= optimizer)
print(model.summary())
model.fit(training_generator, epochs = 20)

# save models - saving encoders individually for inference
os.mkdir(f'../../models/seq2seq+temporal/{model_type}')
model.save(f'../../models//seq2seq+temporal/{model_type}/{model_type}_main.h5')

# save some additional models for inference
enoder_temporal_model = Model(inputs = [x_input, times_in], outputs=[lstm_enc_output, enc_s_state, enc_c_state])
enoder_temporal_model.save(f'../../models/seq2seq+temporal/{model_type}/{model_type}_temporal_enc.h5')



















