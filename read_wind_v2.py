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



model_type ="solar"

plot_temporal_attention = True
plot_spatial_attention = False


if model_type == 'wind':
	dataset_name = 'train_set_V100_wind.hdf5'
elif model_type == 'demand':
	dataset_name = 'dataset_V2_withtimefeatures_Demand.hdf5'
elif model_type == 'solar':
	dataset_name = 'dataset_solar_v30.hdf5'
elif model_type == 'price':
	dataset_name = 'dataset_V2_DAprice.hdf5'


# collect param sizes
f = h5py.File(f"./Data/{model_type}/Processed_Data/{dataset_name}", "r")
features = np.empty_like(f['train_set']['X1_train'][0])
times_in = np.empty_like(f['train_set']['X2_train'][0])
times_out = np.empty_like(f['train_set']['X3_train'][0])
labels = np.empty_like(f['train_set']['y_train'][0])
x_len = f['train_set']['X1_train'].shape[0]
y_len = f['train_set']['y_train'].shape[0]
print('size parameters loaded')

if model_type != "price":
	height, width, channels = features.shape[0], features.shape[1], features.shape[2]
else:
	channels = features.shape[-1]

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

if model_type != "price":
	spatial_enc = load_model(f'./Models/{model_type}_models/q_all/{model_type}Generation_encoderModelSpatial_Q_all.h5') 

# load and process data
f = h5py.File(f"./Data/{model_type}/Processed_Data/{dataset_name}", "r")

set_type = 'test'

X_train1 = f[f'{set_type}_set'][f'X1_{set_type}'][0:3000]
X_train2 = f[f'{set_type}_set'][f'X2_{set_type}'][0:3000]
X_train3 = f[f'{set_type}_set'][f'X3_{set_type}'][0:3000]
X_train4 = f[f'{set_type}_set'][f'X1_{set_type}'][0:3000]
y_train = f[f'{set_type}_set'][f'y_{set_type}'][0:3000]

# get time relevant time references
with open(f'./Data/{model_type}/Processed_Data/time_refs_{model_type}.pkl', 'rb') as time_file:
	time_refs = load(time_file)

input_times = time_refs[f'input_times_{set_type}'][0:3000]
output_times = time_refs[f'output_times_{set_type}'][0:3000]


print(input_times[0])

print(output_times[0])

exit()


time_file.close()  


input_start, output_start = 0, input_seq_size

seqX1, seqX2, seqX3, seqX4, seqY = [], [], [], [], []

times_in, times_out = [], []

# a = np.array(X_train1)

while (output_start + output_seq_size) <= len(y_train):
	# offset handled during pre-processing
	input_end = input_start + input_seq_size
	output_end = output_start + output_seq_size

	# inputs
	seqX1.append(X_train1[input_start:input_end])
	seqX2.append(X_train2[input_start:input_end])
	times_in.append(input_times[input_start:input_end])

	# outputs
	seqX3.append(X_train3[output_start:output_end])
	if model_type != 'price':
		a = X_train4[output_start:output_end][:,:,:,1:]
		a = np.average(a, axis=(1,2))
	else:
		a = X_train4[output_start:output_end][:,1:]
	seqX4.append(a)
	seqY.append(y_train[output_start:output_end])
	times_out.append(output_times[input_start:input_end])

	input_start += output_seq_size
	output_start += output_seq_size


x1, x2, x3, x4, y = np.array(seqX1), np.array(seqX2), np.array(seqX3), np.array(seqX4), np.array(seqY)
times_in, times_out = np.array(times_in), np.array(times_out)
f.close() 


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
	if model_type != "price":
		decoder_input = Input(shape=(1, times_out_dim + (channels-1)))
	else:
		decoder_input = Input(shape=(1, times_out_dim))

	if model_type != 'price':
		enc_out = concatenate([out_nwp, times_out], axis=-1)
	else:
		enc_out = times_out

	# context and previous output
	attn_weights_temp_test, context = model.get_layer(f'temporal_attention_q_{quantile}')(lstm_enc_output_test, enc_out, s_state0, c_state0)

	if model_type != 'price':
		attn_weights_spat_test, context_spat_test = model.get_layer(f'spatial_attention_q_{quantile}')(ccn_enc_output_test, enc_out, s_state0, c_state0)

		# context & previous output combine
		context = concatenate([context, context_spat_test], axis=-1) 

	# decoder_input = concatenate([out_nwp, times_out])

	# Decoder inference
	dec_output, s_state, c_state = model.get_layer(f'decoder_q_{quantile}')(decoder_input, initial_state=[s_state0, c_state0])

	# combine context and prediction
	prediction = concatenate([context, K.expand_dims(dec_output,axis=1)])

	# final dense layer
	pred_test = model.get_layer(f'dense1_q_{quantile}')(prediction)
	pred_test = model.get_layer(f'dense3_q_{quantile}')(pred_test)

	# Inference Model
	if model_type != 'price':
		deoceder_test_model = Model(inputs=[times_in, times_out, out_nwp, decoder_input, ccn_enc_output_test, lstm_enc_output_test, s_state0, c_state0], outputs=[pred_test, s_state, c_state, attn_weights_temp_test, attn_weights_spat_test])  
	else:
		deoceder_test_model = Model(inputs=[times_in, times_out, out_nwp, decoder_input, lstm_enc_output_test, s_state0, c_state0], outputs=[pred_test, s_state, c_state, attn_weights_temp_test])  
	return deoceder_test_model

# dictionary to store decoder models
decoder_models = {}

# instantiate model for each quantile
for q in quantiles:
	decoder_models[f'{q}'] = inference_dec_model(q)

# store predictions
predictions = {}
quantile_temporal_attns = {}



# loop through each sample, passing individually to model
for q in quantiles:
	print(q)

	# set hidden states to zero
	s_state, c_state = s0, c0

	# empty arrays to store all results
	total_pred = np.empty((x1.shape[0], Ty, 1))
	total_temp = np.empty((x1.shape[0], Tx, Ty))
	if model_type != 'price':
		total_spat = np.empty((x1.shape[0], 320, Ty)) # 320 is the fixed spatial attention res

	decoder = decoder_models[f'{q}']

	for idx in range(x1.shape[0]): # loop through each sample, to keep track of hidden states

		# create empty results for results per sample
		outputs = []
		spatial_attns = []
		temporal_attns = []

		# create final inference model
		lstm_enc_output, enc_s_state, enc_c_state = temporal_enc([x1[idx:idx+1], x2[idx:idx+1]])

		if model_type != 'price':
			ccn_enc_output = spatial_enc(x1[idx:idx+1])
			intial_in = np.average(x1[idx:idx+1], axis=(2,3))
		else:
			intial_in = x1[idx:idx+1]

		for ts in range(Ty):

			if model_type != 'price': 
				# declare decoder input 
				if ts > 0:
					decoder_input = concatenate([x4[idx:idx+1,ts-1:ts,:], x3[idx:idx+1,ts-1:ts,:]], axis=-1)
				else:
					decoder_input = concatenate([intial_in[:,-1:,1:], x2[idx:idx+1,-1:,:]], axis=-1)  
			else:
				if ts > 0:
					decoder_input = x3[idx:idx+1,ts-1:ts,:]
				else:
					decoder_input = x2[idx:idx+1,-1:,:]

			if model_type != 'price':  
				pred, s_state, c_state, attn_weights_temp_test, attn_weights_spat_test = decoder([x2[idx:idx+1,ts:ts+1,:], x3[idx:idx+1,ts:ts+1,:], x4[idx:idx+1,ts:ts+1,:], decoder_input, ccn_enc_output, lstm_enc_output, s_state, c_state])
				spatial_attns.append(attn_weights_spat_test)
			else:
				pred, s_state, c_state, attn_weights_temp_test = decoder([x2[idx:idx+1,ts:ts+1,:], x3[idx:idx+1,ts:ts+1,:], x4[idx:idx+1,ts:ts+1,:], decoder_input, lstm_enc_output, s_state, c_state])

			outputs.append(pred)
			temporal_attns.append(attn_weights_temp_test)

		combined_outputs = np.concatenate(outputs, axis=1)
		combined_temp_attn = np.concatenate(temporal_attns, axis=-1)
		
		total_pred[idx, : , :] = combined_outputs
		total_temp[idx, : , :] = combined_temp_attn

		if model_type != 'price':
			combined_spat_attn = np.concatenate(spatial_attns, axis=-1)
			total_spat[idx, : , :] = combined_spat_attn

	predictions[f'{q}'] = total_pred
	quantile_temporal_attns[f'{q}'] = total_temp


plot_ref = 0
idx = 0

# plot predictions
for idx, (key, values) in enumerate(predictions.items()):
	plt.plot(values[plot_ref:plot_ref+7,:].flatten(), label=f"prediction_{key}")

plt.plot(y[plot_ref:plot_ref+7,:,0].flatten(), label="actual")
plt.legend()
plt.show()


# plot temporal attention (quantile 0.5)
att_w_temp = np.transpose(quantile_temporal_attns['0.5'][idx])
if model_type != "price":
	x = np.average(x1, axis=(2,3))[idx, :]
else:
	x = x1[idx, :]

y_attn = y[idx, :, 0]
y_hat = predictions['0.5'][idx, :]

#make attention plotting function
def temporal_attention_graph(x, y, att_w_temp):

	fig = plt.figure(figsize=(24, 8))
	gs = gridspec.GridSpec(ncols=90, nrows=100)

	upper_axis = fig.add_subplot(gs[0:20, 10:75])
	left_axis = fig.add_subplot(gs[25:, 0:8])
	atten_axis = fig.add_subplot(gs[25:, 10:])

	upper_axis.plot(x)
	upper_axis.set_xlim([0, Tx])
	upper_axis.set_ylim([0, 1])
	upper_axis.set_xticks(range(0, Tx))
	upper_axis.set_xticklabels(range(0, Tx))

	left_axis.plot(y, range(0,Ty), label='Prediction')
	left_axis.plot(y_hat, range(0,Ty), label='True')
	left_axis.set_ylim([0, Ty])
	left_axis.set_yticks(range(0, Ty, 6))
	left_axis.set_yticklabels(range(0, Ty, 6))
	left_axis.invert_yaxis()

	sns.heatmap(att_w_temp, cmap='flare', ax = atten_axis, vmin=0, vmax=0.01)
	atten_axis.set_xticks(range(0, Tx))
	atten_axis.set_xticklabels(range(0, Tx))
	atten_axis.set_yticks(range(0, Ty, 4))
	atten_axis.set_yticklabels(range(0, Ty, 4))

	plt.show()


if plot_temporal_attention is True:
	temporal_attention_graph(x, y_attn, att_w_temp)





# plot spatial attention
def plot_spatial_predictions(spatial_data, title, height_scale, width_scale, frame_num):

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
		spatial_over = np.resize(spatial_data[i], (height_scale, width_scale))
		# overlay = ax.imshow(spatial_over, cmap='viridis', alpha=0.5, extent=extent)
		overlay.set_data(spatial_over)
		txt.set_text(f"Timestep: {i}")
		# plt.cla()

		return [overlay, txt]


	animation_ = FuncAnimation(fig, update, frames=frame_num, blit=False, repeat=False)
	plt.show(block=True)	
	# animation_.save(f'{title}_animation.gif', writer='imagemagick')



if plot_spatial_attention is True:
	# transpose spatial attention results
	att_w_spat = np.transpose(total_spat[idx])
	# plot attention weights
	plot_spatial_predictions(att_w_spat, 'Spatial Context', 16, 20, 48)



# add date references to result dictionaries
time_refs = {'input_times': times_in, 'output_times': times_out}

predictions['time_refs'] = time_refs
quantile_temporal_attns['time_refs'] = time_refs 


# save results - forecasted timeseries matrix
with open(f'forecasted_time_series_{model_type}.pkl', 'wb') as ts_file:
	dump(predictions, ts_file)


# save results - forecasted tempotal attention matrix
with open(f'attention_data_{model_type}.pkl', 'wb') as attention_file:
	dump(quantile_temporal_attns, attention_file)














