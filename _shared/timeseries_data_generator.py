import numpy as np
import tensorflow

# as adapted from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(tensorflow.keras.utils.Sequence):

	def __init__(self, dataset_name, x_length, y_length, hidden_states, batch_size, shuffle):
		self.dataset_name = dataset_name
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.n_s = hidden_states
		self.xlen = x_length
		self.ylen = y_length 
		self.index_ref = 0         
		self.on_epoch_end()

	def __len__(self):
		# 'number of batches per Epoch'      
		return int(np.floor((self.ylen - input_seq_size - (output_seq_size-1)) / self.batch_size))

	def __getitem__(self, index):

		# input and output indexes relative current batch size and data generator index reference
		input_indexes = self.input_indexes[(index*self.batch_size) : (index*self.batch_size) + (self.batch_size + (input_seq_size-1))]
		output_indexes = self.output_indexes[(index*self.batch_size) + input_seq_size : (index*self.batch_size) + input_seq_size + (self.batch_size + (output_seq_size-1))]

		# Generate data
		(X_train1, X_train2, X_train3, X_train4, s0, c0), y_train = self.__data_generation(input_indexes, output_indexes)  

		# replicate labels for each quantile
		y_trues = [y_train for i in quantiles]    

		# extend true values for spatial and temporal attention (only relavant if compiled model used for inference)  
		# y_trues.extend([[], []]) 
     
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
        
        # convert to numpy arrays
		seqX1, seqX2, seqX3, seqX4, seqY = np.array(seqX1), np.array(seqX2), np.array(seqX3), np.array(seqX4), np.array(seqY)
		
		return seqX1, seqX2, seqX3, seqX4, seqY

	def __data_generation(self, input_indexes, output_indexes):

		# load data for current batch
		f = h5py.File(f"../../data/processed/{model_type}/{self.dataset_name}", "r")      
		X_train1 = f['train_set']['X1_train'][input_indexes] # main feature array
		X_train2 = f['train_set']['X2_train'][input_indexes] # input time features from feature engineering
		X_train3 = f['train_set']['X3_train'][output_indexes] # output time features from feature engineering

		# no spatial data if model is training for price forecasting
		if model_type != 'price':        
			X_train4 = f['train_set']['X1_train'][output_indexes][:,:,:,1:] # all nwp features apart from the generation itself
			X_train4 = np.average(X_train4, axis=(1,2))
		else: 
			X_train4 = f['train_set']['X1_train'][output_indexes][:,1:]

		y_train = f['train_set']['y_train'][output_indexes]

		f.close()  

        # convert to sequence data
		X_train1, X_train2, X_train3, X_train4, y_train = self.to_sequence(X_train1, X_train2, X_train3, X_train4, y_train)

		s0 = np.zeros((self.batch_size, self.n_s))
		c0 = np.zeros((self.batch_size, self.n_s))

		return (X_train1, X_train2, X_train3, X_train4, s0, c0), y_train