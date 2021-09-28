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
from tensorflow.keras.layers import Input, Activation, AveragePooling2D, Lambda, concatenate, Flatten, BatchNormalization, RepeatVector, Permute, Lambda
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




# create inference object to get multi-quantile predictions
class inference():
	def __init__(self, h5_file_path, dset_type):
		self.h5_file = h5_file_path
		self.dset_type = dset_type # string train or test

		# load input data
		self.file = h5py.File(f"{self.h5_file}", "r")
		self.features = self.file[f'{self.dset_type}_set'][f'X1_{self.dset_type}']
		self.input_times = self.file[f'{self.dset_type}_set'][f'X2_{self.dset_type}']
		self.output_times = self.file[f'{self.dset_type}_set'][f'X3_{self.dset_type}']
		self.y_true = self.file[f'{self.dset_type}_set'][f'y_{self.dset_type}']

		# instaniate shape parameters
		self.tx = self.features.shape[1]
		self.ty = self.y_true.shape[1]
		self.conv_wd = self.features.shape[2]
		self.conv_ht = self.features.shape[3]
		self.Tin = self.input_times.shape[-1]
		self.Tout = self.output_times.shape[-1]
		self.ns = 128 # manually optimised

		# create empty dictionary for storing results
		self.dataset_results = {
			'predictions' : {},
			'temporal_attentions' : {},
			'spatial_attentions' : {}
		}

		# load or create instance for central predictions
		# self._check_for_central_predictions()

	# check for q_0.5 predictions 
	# def _check_for_central_predictions(self):
	# 	file_name = './Results/predictions+attentions.hdf5'
		
	# 	if os.path.isfile(file_name):
	# 		file = h5py.File(file_name, 'r')
	# 		self.central_prediction = file['predictions']['q_0.5']
	# 		file.close()
	# 	else:
	# 		self.central_prediction = np.array([])


	def load_main_model(self, quantile):
		self.model = load_model(f'./Models/solar_models/quantile_{quantile}/solarGeneration_forecast_MainModel_Q_{quantile}.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention})


	def encoder_model(self, quantile):
		enoder_temporal_model = load_model(f'./Models/solar_models/quantile_{quantile}/solarGeneration_encoderModelTemporal_Q_{quantile}.h5')
		enoder_spatial_model = load_model(f'./Models/solar_models/quantile_{quantile}/solarGeneration_encoderModelSpatial_Q_{quantile}.h5')

		return enoder_temporal_model, enoder_spatial_model


	def decoder_model(self, quantile):
		# Encoder outputs for setup
		ccn_enc_output_test = Input(shape=(320, 128))
		lstm_enc_output_test = Input(shape=(self.tx, self.ns)) #+ times_in_dim
		s_state0 = Input(shape=(self.ns,))
		c_state0 = Input(shape=(self.ns,))

		# Decoder Input
		dec_input_test = Input(shape=(None, 1))
		# dec_input_test_int = Input(shape=(1, n_s)) #+ times_in_dim
		times_out_test = Input(shape=(1, self.Tout))

		# context and previous output
		attn_weights_temp_test, context_temp_test = self.model.get_layer('temporal_attention')(lstm_enc_output_test, s_state0, c_state0)
		attn_weights_spat_test, context_spat_test = self.model.get_layer('spatial_attention')(ccn_enc_output_test, s_state0, c_state0)

		# context & previous output combine
		context_test = concatenate([context_spat_test, context_temp_test], axis=-1) 
		dec_input_concat_test = concatenate([context_test, dec_input_test], axis=-1)
		# dec_input_concat_test_int = concatenate([context_test, dec_input_test_int], axis=-1)

		# combine with decoder inputs
		dec_input_concat_test = concatenate([dec_input_concat_test, times_out_test], axis=-1)
		# dec_input_concat_test_int = concatenate([dec_input_concat_test_int, times_out_test], axis=-1)

		dec_output, s_state, c_state = self.model.get_layer('lstm_1')(dec_input_concat_test, initial_state=[s_state0, c_state0])

		pred_test = self.model.get_layer('conv1d')(dec_output)
		pred_test = self.model.get_layer('conv1d_1')(pred_test)
		pred_test = self.model.get_layer('conv1d_2')(pred_test)

		# Inference Model
		deoceder_model = Model(inputs=[dec_input_test, times_out_test, lstm_enc_output_test, ccn_enc_output_test, s_state0, c_state0], outputs=[pred_test, s_state, c_state, attn_weights_temp_test, attn_weights_spat_test]) 

		return deoceder_model


	def predict(self, quantile, enoder_temporal_model, enoder_spatial_model, decoder_model):

		predictions = []
		temporal_attentions = []
		spatial_attentions = []

		# flag if central predictions not made, necessary for TF
		if len(self.central_prediction) == 0 and quantile != 0.5:
			print("central predictions required for all quantile inference")
			exit()

		# inference run
		enc_temp_out, s_state, c_state  = enoder_temporal_model.predict([self.features, self.input_times])
		enc_spat_out = enoder_spatial_model.predict(self.features)

		# intial decoder input
		dec_input = enc_temp_out[:,-1,-1]
		dec_input = K.expand_dims(dec_input, axis=1)
		dec_input = K.expand_dims(dec_input, axis=-1)

		for t in range(1, self.ty+1):
			# get current 'times out' reference
			times_out_single = self.output_times[:,t-1,:]
			times_out_single = K.expand_dims(times_out_single, axis=1)

			prediction, s_state, c_state, temporal_attention, spatial_attention = decoder_model.predict([dec_input, times_out_single, enc_temp_out, enc_spat_out, s_state, c_state])
			
			# store prediction from q_0.5
			# if quantile == 0.5:
			dec_input = prediction
			# else:
			# 	dec_input = self.central_prediction[:,t-1,:]

		    
			if t == 1:
				predictions = prediction
				temporal_attentions = temporal_attention
				spatial_attentions = spatial_attention
			else:
				predictions = np.concatenate([predictions, prediction], axis=1)
				temporal_attentions = np.concatenate([temporal_attentions, temporal_attention], axis=-1)
				spatial_attentions = np.concatenate([spatial_attentions, spatial_attention], axis=-1)


		# if len(self.central_prediction) == 0 and quantile == 0.5:
		# 	self.central_prediction = predictions


		return predictions, temporal_attentions, spatial_attentions



	def save_predictions(self, qunatile, predictions, spatial_attentions):

		self.dataset_results['predictions'][f'q_{quantile}'] = predictions
		self.dataset_results['temporal_attentions'][f'q_{quantile}'] = temporal_attentions
		self.dataset_results['spatial_attentions'][f'q_{quantile}'] = spatial_attentions

		f = h5py.File('./Results/predictions+attentions.hdf5', 'w')

		for group_name in self.dataset_results:
			group = f.create_group(group_name)
			for dset_name in self.dataset_results[group_name]:
				dset = group.create_dataset(dset_name, data = self.dataset_results[group_name][dset_name])
		f.close()


	def evaluate_predictions(self, lower_q, upper_q):
		'''
		Theory from Bazionis & Georgilakis (2021): https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiUprb39qbyAhXNgVwKHWVsA50QFnoECAMQAQ&url=https%3A%2F%2Fwww.mdpi.com%2F2673-4826%2F2%2F1%2F2%2Fpdf&usg=AOvVaw1AWP-zHuNGrw8pgDfUS09e
		func to caluclate probablistic forecast performance
		Prediction Interval Coverage Probability (PICP)
		Prediction Interval Nominla Coverage (PINC)
		Average Coverage Error (ACE) [PICP - PINC]
		'''
		test_len = len(self.y_true)

		y_true = self.y_true.ravel()
		lower_pred = self.dataset_results['predictions'][f'q_{lower_q}'].ravel()
		upper_pred = self.dataset_results['predictions'][f'q_{upper_q}'].ravel()

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

		# create pandas df
		metrics = pd.DataFrame({'PICP': picp, 'PINC': pinc, 'ACE': ace, 'PINAW': pinaw, 'PINRW': pinrw}, index={alpha})
		metrics.index.name = 'Prediction_Interval'

		print(metrics)

		return metrics



	def plot_predictions(self, index, daynum):

		for idx, q in enumerate(self.dataset_results['predictions']):
			plt.plot(self.dataset_results['predictions'][f'{q}'][index:index+daynum, :, :].flatten(), color='tab:red', linestyle='--')

		print(self.dataset_results['predictions'].keys())

		plt.plot(self.y_true[index:index+daynum, :, 0].flatten(), 'k')
		plt.show()


	def plot_temporal_predictions(self, index, quantile):

		fig = plt.figure(figsize=(24, 8))
		gs = gridspec.GridSpec(ncols=90, nrows=100)

		upper_axis = fig.add_subplot(gs[0:20, 10:75])
		left_axis = fig.add_subplot(gs[25:, 0:8])
		atten_axis = fig.add_subplot(gs[25:, 10:])

		upper_axis.plot(np.average(self.features, axis=(2,3))[index, :])
		upper_axis.set_xlim([0, self.tx])
		upper_axis.set_ylim([0, 0.5])
		upper_axis.set_xticks(range(0, self.tx))
		upper_axis.set_xticklabels(range(0, self.tx))

		left_axis.plot(self.y_true[index, :, 0], range(0,self.ty), label='True')
		left_axis.plot(self.dataset_results['predictions'][f'q_{quantile}'], range(0,self.ty), label='Prediction')
		left_axis.set_ylim([0, self.ty])
		left_axis.set_yticks(range(0, self.ty, 4))
		left_axis.set_yticklabels(range(0, self.ty, 4))
		left_axis.invert_yaxis()
		left_axis.legend()
	    
		sns.heatmap(np.transpose(self.dataset_results['temporal_attentions'][index]), cmap='flare', ax = atten_axis)
		atten_axis.set_xticks(range(0, self.tx))
		atten_axis.set_xticklabels(range(0, self.tx))
		atten_axis.set_yticks(range(0, self.ty, 4))
		atten_axis.set_yticklabels(range(0, self.ty, 4))

		plt.show()


	def plot_spatial_predictions(self, spatial_data, title, height_scale, width_scale, frame_num):

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



quantiles = [0.5, 0.1, 0.9]

# inference model inputs
dataset_name = "train_set_V10_withtimefeatures_120hrinput_float32.hdf5"
h5_file_path = f"./Data/solar/Processed_Data/{dataset_name}"
dset_type = "train" # train or test string


# instaniate inference class
inference = inference(h5_file_path, dset_type)

predictions_all = {}
temporal_attentions_all = {} 
spatial_attentions_all = {}


# loop for each quantile
for quantile in quantiles:
	print(f'testing q_{quantile} model...')

	# load main model
	inference.load_main_model(quantile)

	# load encoders models for current quantile
	enoder_temporal_model, enoder_spatial_model = inference.encoder_model(quantile)

	# load  main model for current quantile
	decoder_model = inference.decoder_model(quantile)

	# run predictions for quantile
	predictions, temporal_attentions, spatial_attentions = inference.predict(quantile, enoder_temporal_model, enoder_spatial_model, decoder_model)

	# save predictions in h5py
	inference.save_predictions(quantile, predictions, spatial_attentions)

# plot predictions
inference.plot_predictions(index=850, daynum=5)














