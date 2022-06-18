import keras
from keras.models import load_model, model_from_json
from keras.backend import sigmoid
from tensorflow.keras.layers import Input, Activation, concatenate, Lambda
import numpy as np
import h5py
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from pickle import load, dump
import matplotlib.pyplot as plt
from keras.utils.generic_utils import get_custom_objects


from keras.utils.generic_utils import get_custom_objects# import custom classes 
from _shared.attention_layer import attention

# script to produce test-set predictions for Bi-directional LSTM model

# declare model type
model_type = 'seq2seq+temporal' # - bilstm, seq2seq, seq2seq+temporal

# indicate model type 
forecast_var = 'price'

# quantiles - needed for key references - ensure aligns with trained model
quantiles = [0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95]

# define swish function for use within comptile model
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
  
# Below in place of swish you can take any custom key for the name 
get_custom_objects().update({'swish': Activation(swish)})

# load trainined model
if model_type != 'seq2seq+temporal':
	model = load_model(f'../../Models/{model_type}/{forecast_var}/q_all_{model_type}/{forecast_var}_{model_type}.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f)})
else:
	model = load_model(f'../../Models/{model_type}/{forecast_var}/q_all_{model_type}/{forecast_var}_{model_type}.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention, 'Activation': Activation(swish)})

# load time references
with open(f'../../data/processed/{forecast_var}/time_refs_{forecast_var}_v2.pkl', 'rb') as time_file:
	time_refs = load(time_file)

input_times = time_refs[f'input_times_test']
output_times = time_refs[f'output_times_test']

time_file.close()  

# load and process data
f = h5py.File(f"../../data/processed/{forecast_var}/dataset_{forecast_var}_v2.hdf5", "r")

set_type = 'test'
X_train1 = f[f'{set_type}_set'][f'X1_{set_type}']
X_train2 = f[f'{set_type}_set'][f'X2_{set_type}']
X_train3 = f[f'{set_type}_set'][f'X3_{set_type}']
X_train4 = f[f'{set_type}_set'][f'X1_{set_type}']
y_train = f[f'{set_type}_set'][f'y_{set_type}']

input_seq_size = 336
output_seq_size = 48

input_start, output_start = 0, input_seq_size

seqX1, seqX2, seqX3, seqX4, seqY = [], [], [], [], []

times_in, times_out = [], []

# sequence the data
while (output_start + output_seq_size) <= len(y_train):
	# offset handled during pre-processing
	input_end = input_start + input_seq_size
	output_end = output_start + output_seq_size

	# inputs
	seqX1.append(X_train1[input_start:input_end])
	seqX2.append(X_train2[input_start:input_end])

	times_in.append(input_times[input_start:input_end])

	# outputs
	seqY.append(y_train[output_start:output_end])
	seqX3.append(X_train3[output_start:output_end])
	seqX4.append(X_train4[output_start:output_end])
	times_out.append(output_times[output_start:output_end])

	input_start += output_seq_size
	output_start += output_seq_size

x1, x2, x3, x4, y = np.array(seqX1), np.array(seqX2), np.array(seqX3), np.array(seqX4), np.array(seqY)
times_in, times_out = np.array(times_in), np.array(times_out)

f.close() 

# load scaler 
scaler = load(open(f'../../data/processed/{forecast_var}/_scaler/scaler_{forecast_var}_v2.pkl', 'rb'))

# average inputs over spatial dimensions
if forecast_var != 'price':
	if model_type != 'seq2seq+temporal': 
		x1 = np.average(x1, axis=(2,3))

	x4 = np.average(x4, axis=(2,3))
	x4 = x4[:,:,1:]
else:
	x4 = x4[:,:,:-1]

# cache test set length
test_len = y.shape[0]

# delcare intial hidden states
s0 = np.zeros((y.shape[0], 32,))
c0 = np.zeros((y.shape[0], 32,))


print('predicting')
if model_type == 'bilstm':
	results = model.predict([x1, x2])
elif model_type == 'seq2seq+temporal':
	results = model.predict([x1, x2, x3, x4, s0, c0])
	quantile_temporal_attns = results[-1]
else:
	results = model.predict([x1, x2, x3, x4])


results_dict = {}

# inverse transform predictions + transfer to dictionary
for idx in range(len(quantiles)):
	results_dict[str(quantiles[idx])] = scaler.inverse_transform(results[idx].reshape(-1,1)).reshape(test_len, output_seq_size, 1)

# inverse transform true values 
y_true = scaler.inverse_transform(y.reshape(-1,1)).reshape(test_len, output_seq_size, 1)

# create time_refs dictionary
times_refs = {'input_times': times_in, 'output_times': times_out}

# create results dictionary for performance analysis / plotting
results_dict['time_refs'] = times_refs
results_dict['y_true'] = y_true

print(results_dict.keys())

# save results - forecasted timeseries matrix
with open(f'../../results/{forecast_var}/{model_type}/forecasted_time_series_{forecast_var}_{model_type}.pkl', 'wb') as ts_file:
	dump(results_dict, ts_file)

# save results - forecasted tempotal attention matrix
if (model_type == 'seq2seq+temporal') and (forecast_var == 'price'):

	# construct attention results dictionary
	attention_results = {}
	attention_results['0.5'] = quantile_temporal_attns
	attention_results['time_refs'] = times_refs
	attention_results['input_features'] = x1

	with open(f'../../results/{forecast_var}/{model_type}/attention_data_{forecast_var}_seq2seq+temporal.pkl', 'wb') as attention_file:
		dump(attention_results, attention_file)





