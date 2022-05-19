import keras
from keras.models import load_model, model_from_json
import numpy as np
import h5py
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from pickle import load, dump
import matplotlib.pyplot as plt


# script to produce test-set predictions for Bi-directional LSTM model

# indicate model type 
model_type = 'solar'

# quantiles - needed for key references - ensure aligns with trained model
quantiles = [0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95]

# load trainined model
model = load_model(f'../../Models/bilstm/{model_type}/q_all_bilstm/{model_type}_bilstm.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f)})

# load time references
with open(f'../../data/processed/{model_type}/time_refs_{model_type}_v2.pkl', 'rb') as time_file:
	time_refs = load(time_file)

input_times = time_refs[f'input_times_test']
output_times = time_refs[f'output_times_test']

time_file.close()  

# load and process data
f = h5py.File(f"../../data/processed/{model_type}/dataset_{model_type}_v2.hdf5", "r")

set_type = 'test'
X_train1 = f[f'{set_type}_set'][f'X1_{set_type}']
X_train2 = f[f'{set_type}_set'][f'X2_{set_type}']
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
	times_out.append(output_times[output_start:output_end])

	input_start += output_seq_size
	output_start += output_seq_size

x1, x2, y = np.array(seqX1), np.array(seqX2), np.array(seqY)
times_in, times_out = np.array(times_in), np.array(times_out)

f.close() 

# load scaler 
scaler = load(open(f'../../data/processed/{model_type}/_scaler/scaler_{model_type}_v2.pkl', 'rb'))

# average inputs over spatial dimensions
x1 = np.average(x1, axis=(2,3))

# cache test set length
test_len = y.shape[0]

print('predicting')
results = model.predict([x1, x2])

results_dict = {}

# inverse transform predictions + transfer to dictionary
for idx in range(len(results)):
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
with open(f'../../results/{model_type}/bilstm/forecasted_time_series_{model_type}_bilstm.pkl', 'wb') as ts_file:
	dump(results_dict, ts_file)






