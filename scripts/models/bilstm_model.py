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
params = {'batch_size': 64,
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

		input_indexes = self.input_indexes[(index*self.batch_size) : (index*self.batch_size) + (self.batch_size + (input_seq_size-1))]
		output_indexes = self.output_indexes[(index*self.batch_size) + input_seq_size : (index*self.batch_size) + input_seq_size + (self.batch_size + (output_seq_size-1))]

		# Generate data
		(X_train1, X_train2), y_train = self.__data_generation(input_indexes, output_indexes)  

		y_trues = [y_train for i in quantiles]    

		return (X_train1, X_train2), (y_trues) # pass empty training outputs to extract extract attentions

	def on_epoch_end(self):
		# set length of indexes for each epoch
		self.input_indexes = np.arange(self.xlen)
		self.output_indexes = np.arange(self.ylen)
 
		if self.shuffle == True:
			np.random.shuffle(self.input_indexes)

	def to_sequence(self, x1, x2, y):
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
			seqY.append(y[output_start:output_end])

			input_start += 1  
			output_start += 1
            
		seqX1, seqX2, seqY = np.array(seqX1), np.array(seqX2), np.array(seqY)
		
		return seqX1, seqX2, seqY

	def __data_generation(self, input_indexes, output_indexes):

		f = h5py.File(f"../../data/processed/{model_type}/{self.dataset_name}", "r")      

		X_train2 = f['train_set']['X2_train'][input_indexes]

		if model_type != 'price':        
			X_train1 = f['train_set']['X1_train'][input_indexes][:,:,:,:]
			X_train1 = np.average(X_train1, axis=(1,2))
		else: 
			X_train1 = f['train_set']['X1_train'][input_indexes][:,:]
 

		y_train = f['train_set']['y_train'][output_indexes]
		# decoder_input = f['train_set']['y_train'][output_indexes]
		f.close()  

        # convert to sequence data
		X_train1, X_train2, y_train = self.to_sequence(X_train1, X_train2, y_train)

  
		return (X_train1, X_train2), y_train

training_generator = DataGenerator(dataset_name = dataset_name, x_length = x_len, y_length = y_len,  **params)

###########################################_____MODEL_ARCHITECTURE_____#################################################

# cpature some more useful dimensions
Tx = input_seq_size
Ty = output_seq_size

channels = features.shape[-1]

times_in_dim = times_in.shape[-1]
times_out_dim = times_out.shape[-1]

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
quantile_predictions = []

for q in quantiles: 

	combined_inputs = concatenate([x_input, times_in], axis=-1, name=f'concat_q_{q}')

	layer1, _, _, _, _ = Bidirectional(LSTM(32, return_sequences = False, return_state = True), name=f'biLSTM_q_{q}')(combined_inputs)
	layer2 = Dense(48, name=f'dense1_q_{q}')(layer1)

	if model_type == 'solar':
		layer2 = Activation('relu', name=f'relu_act_q_{q}')(layer2)

	quantile_predictions.append(layer2)

model = Model(inputs = [x_input, times_in], outputs = quantile_predictions)


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
os.mkdir(f'../../models/bilstm/{model_type}')
model.save(f'../../models/bilstm/{model_type}/{model_type}_bilstm.h5')



















