import pandas as pd
import numpy as np
import os 
import sys
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import load_model
from keras import Model
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Activation, concatenate, Lambda
from tensorflow.keras.layers import Reshape
from keras.callbacks import ModelCheckpoint
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from pickle import load
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import mean_absolute_error, mean_squared_error
import h5py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import seaborn as sns
from pickle import dump, load

import geopandas
import contextily as ctx

# import custom classes 
from _shared.attention_layer import attention



# choose model type to run test for
model_type ="wind"

# declare dataset file
dataset_name = f'dataset_{model_type}.hdf5'

# choose to activate plot functions
plot_temporal_attention = False 
plot_spatial_attention = False

# declare quantiles
quantiles = [0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95]

# index to declare which test result to plot
plot_ref = 0

# load scaler 
scaler = load(open(f'../../data/processed/{model_type}/_scaler/scaler_{model_type}.pkl', 'rb'))

# collect param sizes
f = h5py.File(f"../../data/processed/{model_type}/{dataset_name}", "r")
features = np.empty_like(f['train_set']['X1_train'][0])
times_in = np.empty_like(f['train_set']['X2_train'][0])
times_out = np.empty_like(f['train_set']['X3_train'][0])
labels = np.empty_like(f['train_set']['y_train'][0])
x_len = f['train_set']['X1_train'].shape[0]
y_len = f['train_set']['y_train'].shape[0]
print('size parameters loaded')

# additional params dependent on wether spatial data is present
if model_type != "price":
	height, width, channels = features.shape[0], features.shape[1], features.shape[2]
else:
	channels = features.shape[-1]

times_in_dim = times_in.shape[-1]
times_out_dim = times_out.shape[-1]

# decalre additional usefule params
Tx = 336
Ty = 48
n_s = 32
input_seq_size = Tx
output_seq_size = Ty

# define swish function for use within comptile model
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
  
# Below in place of swish you can take any custom key for the name 
get_custom_objects().update({'swish': Activation(swish)})

# load main model
model = load_model(f'../../models/{model_type}/{model_type}_main.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention, 'Activation': Activation(swish)})

# read encoder models - igoring the spatail encoder in the price forecasting case
temporal_enc = load_model(f'../../models/{model_type}/{model_type}_temporal_enc.h5') 

if model_type != "price":
	spatial_enc = load_model(f'../../models/{model_type}/{model_type}_spatial_enc.h5') 

# load and process data
f = h5py.File(f"../../data/processed/{model_type}/{dataset_name}", "r")

# load test or train data - too much memory to load all data, so just load segment
set_type = 'test'
X_train1 = f[f'{set_type}_set'][f'X1_{set_type}'][0:3000]
X_train2 = f[f'{set_type}_set'][f'X2_{set_type}'][0:3000]
X_train3 = f[f'{set_type}_set'][f'X3_{set_type}'][0:3000]
X_train4 = f[f'{set_type}_set'][f'X1_{set_type}'][0:3000]
y_train = f[f'{set_type}_set'][f'y_{set_type}'][0:3000]

# get time relevant time references
with open(f'../../data/processed/{model_type}/time_refs_{model_type}.pkl', 'rb') as time_file:
	time_refs = load(time_file)

input_times = time_refs[f'input_times_{set_type}'][0:3000]
output_times = time_refs[f'output_times_{set_type}'][0:3000]

time_file.close()  

# begin sequencing of data 
input_start, output_start = 0, input_seq_size

seqX1, seqX2, seqX3, seqX4, seqY = [], [], [], [], []

times_in, times_out = [], []

while (output_start + output_seq_size) <= len(y_train):
	# increment indexes for windowing of data
	input_end = input_start + input_seq_size
	output_end = output_start + output_seq_size

	# inputs
	seqX1.append(X_train1[input_start:input_end])
	seqX2.append(X_train2[input_start:input_end])
	times_in.append(input_times[input_start:input_end])

	# outputs
	seqX3.append(X_train3[output_start:output_end])
	if model_type != 'price':
		nwp_data = X_train4[output_start:output_end][:,:,:,1:]
		nwp_data = np.average(nwp_data, axis=(1,2))
	else:
		nwp_data = X_train4[output_start:output_end][:,1:]
	seqX4.append(nwp_data)
	seqY.append(y_train[output_start:output_end])
	times_out.append(output_times[output_start:output_end])

	input_start += output_seq_size
	output_start += output_seq_size

# make sure all are numpy arrays
x1, x2, x3, x4, y = np.array(seqX1), np.array(seqX2), np.array(seqX3), np.array(seqX4), np.array(seqY)
times_in, times_out = np.array(times_in), np.array(times_out)
f.close() 

# scale actual values 
y_idx = y.shape[0]
y = scaler.inverse_transform(y.reshape(-1,1)).reshape(y_idx, Ty, 1)

# declare intial hidden states
s0 = np.zeros((1, n_s))
c0 = np.zeros((1, n_s))

# function for inference decoder model - one for each quantile
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

	# define input for encoder
	if model_type != 'price':
		enc_in = concatenate([out_nwp, times_out], axis=-1)
	else:
		enc_in = times_out

	# context and previous output
	attn_weights_temp_test, context = model.get_layer(f'temporal_attention_q_{quantile}')(lstm_enc_output_test, enc_in, s_state0, c_state0)

	if model_type != 'price':
		attn_weights_spat_test, context_spat_test = model.get_layer(f'spatial_attention_q_{quantile}')(ccn_enc_output_test, enc_in, s_state0, c_state0)

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

	if model_type == "solar":
		pred_test = model.get_layer(f'relu_act_q_{quantile}')(pred_test)

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
quantile_spatial_attns = {}

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
		combined_spat_attn = np.concatenate(spatial_attns, axis=-1)
		
		total_pred[idx, : , :] = scaler.inverse_transform(combined_outputs[0,:,:])
		total_temp[idx, : , :] = combined_temp_attn

		if model_type != 'price':
			combined_spat_attn = np.concatenate(spatial_attns, axis=-1)
			total_spat[idx, : , :] = combined_spat_attn

	predictions[f'{q}'] = total_pred
	quantile_temporal_attns[f'{q}'] = total_temp
	quantile_spatial_attns[f'{q}'] = total_spat

# plot predictions for specified index
for idx, (key, values) in enumerate(predictions.items()):
	plt.plot(values[plot_ref:plot_ref+7,:].flatten(), label=f"prediction_{key}")

plt.plot(y[plot_ref:plot_ref+7,:,0].flatten(), label="actual")
plt.legend()
plt.show()


# plot temporal attention (quantile 0.5)
att_w_temp = np.transpose(quantile_temporal_attns['0.5'][plot_ref])
if model_type != "price":
	x = np.average(x1, axis=(2,3))[plot_ref, :]
else:
	x = x1[plot_ref, :]

y_attn = y[plot_ref, :, 0]
y_hat = predictions['0.5'][plot_ref, :]

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

	sns.heatmap(att_w_temp, cmap='flare', ax = atten_axis, vmin=0, vmax=0.001)
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
	att_w_spat = np.transpose(total_spat[plot_ref])
	# plot attention weights
	plot_spatial_predictions(att_w_spat, 'Spatial Context', 16, 20, 48)

# function to evaluate quantile performance
def evaluate_predictions(predictions):
	'''
	Theory from Bazionis & Georgilakis (2021): https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiUprb39qbyAhXNgVwKHWVsA50QFnoECAMQAQ&url=https%3A%2F%2Fwww.mdpi.com%2F2673-4826%2F2%2F1%2F2%2Fpdf&usg=AOvVaw1AWP-zHuNGrw8pgDfUS09e
	func to caluclate probablistic forecast performance
	Prediction Interval Coverage Probability (PICP)
	Prediction Interval Nominla Coverage (PINC)
	Average Coverage Error (ACE) [PICP - PINC]
	'''
	test_len = len(predictions['y_true'])

	y_true = predictions['y_true'].ravel()
	lower_pred = predictions[list(predictions.keys())[0]].ravel()
	upper_pred = predictions[list(predictions.keys())[-1]].ravel()
	central_case = predictions[0.5].ravel()

	alpha = upper_pred - lower_pred

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

	# calculate MAE & RMSE
	mae = mean_absolute_error(y_true, central_case)
	rmse = mean_squared_error(y_true, central_case)

	# create pandas df
	metrics = pd.DataFrame({'PICP': picp, 'PINC': pinc, 'ACE': ace, 'PINAW': pinaw, 'PINRW': pinrw, 'MAE': mae, 'RMSE': rmse}, index={alpha})
	metrics.index.name = 'Prediction_Interval'

	print(metrics)

	return eval_metrics



# add date references to result dictionaries
time_refs = {'input_times': times_in, 'output_times': times_out}

predictions['time_refs'] = time_refs
quantile_temporal_attns['time_refs'] = time_refs 

# add x-input data 
quantile_temporal_attns['input_features'] = x1

# add true value for reference to prediction dictionary
predictions['y_true'] = y

# performance evaluation
# evaluate_predictions(predictions)


# save results - forecasted timeseries matrix
with open(f'forecasted_time_series_{model_type}.pkl', 'wb') as ts_file:
	dump(predictions, ts_file)

# save results - forecasted temporal attention matrix
with open(f'attention_data_{model_type}.pkl', 'wb') as attention_file:
	dump(quantile_temporal_attns, attention_file)

# save results - forecasted spatial attention matrix
with open(f'attention_data_{model_type}.pkl', 'wb') as spatial_file:
	dump(quantile_spatial_attns, spatial_file)











