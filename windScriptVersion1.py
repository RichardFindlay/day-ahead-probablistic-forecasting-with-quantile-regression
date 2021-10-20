import pandas as pd
import numpy as np
import netCDF4
from netCDF4 import Dataset
import os 
import glob
import sys
import keras
from collections import OrderedDict
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scipy
from keras import Model
from keras.layers import Input, concatenate, Bidirectional, ConvLSTM2D, Flatten, Dropout, Dense, BatchNormalization, MaxPooling3D, TimeDistributed, Flatten, RepeatVector, LSTM, Conv1D
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import matplotlib.pyplot as plt
import datetime
# import psutil
from pickle import dump, load
import time
import tensorflow as tf

import h5py

# os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #hide tensorflow error 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

np.set_printoptions(threshold=sys.maxsize)

###########################################_____LOAD & PRE-PROCESS DATA_____###########################################

#cache current working directory of main script
workingDir = os.getcwd()

def ncExtract(directory): #will append files if multiple present

	#intialising parameters
	os.chdir(directory)
	files = []
	readVariables = {}
	consistentVars = ['longitude', 'latitude', 'time']

	#read files in directory
	for file in glob.glob("*.nc"):
		files.append(file)
		files.sort()
	
	for i, file in enumerate(files):
		print(file)
		#read nc file using netCDF4
		ncfile = Dataset(file) 
		varaibles = list(ncfile.variables.keys())
		#find unique vars 
		uniqueVars = list(set(varaibles) - set(consistentVars))

		#iteriate and concat each unique variable
		for variable in uniqueVars:

			if i == 0:
				readVariables['data'] = np.empty([0,ncfile.variables['latitude'].shape[0],
					ncfile.variables['longitude'].shape[0]])

			readVar = ncfile.variables[variable][:]

			readVariables['data'] = np.concatenate([readVariables['data'],readVar])

		#read & collect time
		if i == 0:
			readVariables['time'] = np.empty([0])
		
		timeVar = ncfile.variables['time']
		datesVar = netCDF4.num2date(timeVar[:], timeVar.units, timeVar.calendar)
		readVariables['time'] = np.concatenate([readVariables['time'],datesVar])

	#read lat and long
	readVariables['latitude'] = ncfile.variables['latitude'][:]
	readVariables['longitude'] = ncfile.variables['longitude'][:]

	#close ncfile file
	Dataset.close(ncfile)

	#change directory back
	os.chdir(workingDir)

	#define name of extracted data
	fileNameLoc = directory.rfind('/') + 1
	fileName = str(directory[fileNameLoc:])

	return readVariables



#function to filter irregular values out
def lv_filter(data):
	#define +ve and -ve thresholds
	filter_thres_pos = np.mean(np.mean(data)) * (10**(-10))
	filter_thres_neg = filter_thres_pos * (-1)

	#filter data relevant to thresholds
	data[(filter_thres_neg <= data) & (data <= filter_thres_pos)] = 0

	return data


# function to convert 24hr input to 48hrs
def interpolate_4d(array):
	interp_array = np.empty((array.shape[0]*2 , array.shape[1], array.shape[2], array.shape[3]))
	for ivar in range(array.shape[-1]):
		for interp_idx in range(interp_array.shape[0]):
			if (interp_idx % 2 == 0) or (int(np.ceil(interp_idx/2)) == array.shape[0]): 
				interp_array[interp_idx, :, :, ivar] = array[int(np.floor(interp_idx/2)), :, :, ivar]
			else:
				interp_array[interp_idx, :, :, ivar] = (array[int(np.floor(interp_idx/2)), :, :, ivar] + array[int(np.ceil(interp_idx/2)), :, :, ivar]) / 2

	return interp_array


# function to interpolate time
def interpolate_time(time_array):
	interp_time = np.linspace(time_array[0], time_array[-1], len(time_array)*2)

	return interp_time


# function to check for missing nans - if so delete day
def remove_nan_days(x_in, y_out): # assume both are
	# check for missing vals in outputs
	idx = 0
	for i in range(len(y_out)):
		if y_out[idx].isnull().values.any() or x_in[idx].isnull().values.any():
			del x_in[idx]
			del y_out[idx]
			idx -= 1
		idx += 1 

	return x_in, y_out




def format_data_into_timesteps(X1, X2, X3, Y, input_seq_size, output_seq_size, input_times_reference, output_times_reference):
	print('formating data into timesteps & interpolating input data')

	#number of timesteps to be included in each sequence
	seqX1, seqX2, seqX3, seqY_in, seqY, in_times, out_times = [], [], [], [], [], [], []
	input_start, input_end = 0, 0
	output_start = input_seq_size + output_seq_size 
	# output_start = int(input_seq_size*2) - output_seq_size
	# input_seq_size - output_seq_size - nested
	# input_start + input_seq_size - ahead

	while (output_start + output_seq_size) < len(X1):
		# (input_start + input_seq_size + output_seq_size)
		# (input_start + input_seq_size) < len(X1)

		x1 = np.empty((input_seq_size , X1.shape[1], X1.shape[2], X1.shape[3]))
		x2 = np.empty((input_seq_size , X2.shape[1]))
		x3 = np.empty((output_seq_size , X3.shape[1]))
		y_in = np.empty(((input_seq_size), 1))
		y = np.empty((output_seq_size, 1))

		in_time = np.empty(((input_seq_size)), dtype = 'datetime64[ns]')
		out_time = np.empty(((output_seq_size)), dtype = 'datetime64[ns]')

		#define sequences
		input_end = input_start + input_seq_size
		output_end = output_start + output_seq_size

		#add condition to ommit any days with nan values
		if np.isnan(X1[input_start:input_end]).any() == True or np.isnan(X2[input_start:input_end]).any() == True or np.isnan(Y[input_start:input_end]).any() == True:
			input_start += input_seq_size 
			output_start += input_seq_size 
			continue
		elif np.isnan(X3[output_start:output_end]).any() == True or np.isnan(Y[output_start:output_end]).any() == True:
			input_start += output_seq_size 
			output_start += output_seq_size 
			continue

		# #define sequences
		# input_end = input_start + input_seq_size
		# output_end = output_start + output_seq_size

		# #add condition to ommit any days with nan values
		# if np.isnan(X1[input_start:input_end]).any() == True or np.isnan(X2[input_start:input_end]).any() == True or np.isnan(Y[input_start:input_end]).any() == True:
		# 	input_start += input_seq_size 
		# 	output_start += int(input_seq_size *2)
		# 	continue
		# elif np.isnan(X3[output_start:output_end]).any() == True or np.isnan(Y[output_start:output_end]).any() == True:
		# 	input_start += int(output_seq_size /2)
		# 	output_start += output_seq_size 
		# 	continue

		x1[:,:,:,:] = X1[input_start:input_end]
		seqX1.append(x1)
		x2[:,:] = X2[input_start:input_end]
		seqX2.append(x2)
		x3[:,:] = X3[output_start:output_end]
		seqX3.append(x3)
		y_in[:,:] = Y[input_start:input_end]
		# y_in[-48:,:] = 0 # elinimate metered output - only NWP available for prediction day
		seqY_in.append(y_in)
		y[:] = Y[output_start:output_end]
		seqY.append(y)

		in_time[:] = np.squeeze(input_times_reference[input_start:input_end])
		in_times.append(in_time)
		out_time[:] = np.squeeze(output_times_reference[output_start:output_end])
		out_times.append(out_time)
		

		# input_start += output_seq_size  # divide by 2 to compensate for 24hr period (edited)
		# output_start += output_seq_size

		input_start += 1  # divide by 2 to compensate for 24hr period (edited)
		output_start += 1

		# input_start += int(output_seq_size / 2)  # divide by 2 to compensate for 24hr period (edited)
		# output_start += output_seq_size

	print('converting to float32 numpy arrays')
	seqX1 = np.array(seqX1, dtype=np.float32)
	seqX2 = np.array(seqX2, dtype=np.float32)
	seqX3 = np.array(seqX3, dtype=np.float32)
	seqY_in = np.array(seqY_in, dtype=np.float32)
	seqY = np.array(seqY, dtype=np.float32)


	# stack 'Y_inputs' onto the spatial array
	print('combining feature array with lagged outputs')
	broadcaster = np.ones((seqX1.shape[0], seqX1.shape[1], seqX1.shape[2], seqX1.shape[3],  1), dtype=np.float32)
	broadcaster = broadcaster * np.expand_dims(np.expand_dims(seqY_in, axis =2), axis=2)
	seqX1 = np.concatenate((broadcaster, seqX1), axis = -1)


	#split data for train and test sets
	test_set_percentage = 0.1
	test_split = int(len(seqX1) * (1 - test_set_percentage))


	dataset = {
		'train_set' : {
			'X1_train': seqX1[:test_split],
			'X2_train': seqX2[:test_split], # input time features
			'X3_train': seqX3[:test_split], # output time features
			'y_train': seqY[:test_split] 
			},
		'test_set' : {
			'X1_test': seqX1[test_split:],
			'X2_test': seqX2[test_split:], 
			'X3_test': seqX3[test_split:],
			'y_test': seqY[test_split:] 
			}
		# 'time_refs' : {
		# 	'input_times_train': in_times[:test_split],
		# 	'input_times_test': in_times[test_split:], 
		# 	'output_times_train': out_times[:test_split],
		# 	'output_times_test': out_times[test_split:]
		# 	}
	}

	# #create dictionary for training data
	# train_set = {
	# 	'X1_train': seqX1[:test_split],
	# 	'X2_train': seqX2[:test_split], # input time features
	# 	'X3_train': seqX3[:test_split], # output time features
	# 	# 'y_in_train': seqY_in[:test_split],
	# 	'y_train': seqY[:test_split] 
	# }

	# #create dictionary for testing data
	# test_set = {
	# 	'X1_test': seqX1[test_split:],
	# 	'X2_test': seqX2[test_split:], 
	# 	'X3_test': seqX3[test_split:],
	# 	# 'y_in_test': seqY_in[(test_split + 1):], 
	# 	'y_test': seqY[test_split:] 
	# }

	#create dictionary for time references
	time_refs = {
		'input_times_train': in_times[:test_split],
		'input_times_test': in_times[test_split:], 
		'output_times_train': out_times[:test_split],
		'output_times_test': out_times[test_split:]
	}

	return dataset, time_refs
	# train_set, test_set, time_refs



#function to process data in train and test sets
def data_processing(filepaths, labels, input_seq_size, output_seq_size):

	#get dictionary keys
	keys = list(filepaths.keys())

	#dictionaries for extracted vars
	vars_extract = {}
	vars_extract_filtered = {}
	vars_extract_filtered_masked = {}
	vars_extract_filtered_masked_norm = {}

	#define daylight hours mask - relative to total solar radiation 
	# solar_rad_reference = ncExtract('./Data/solar/Raw_Data/Net_Solar_Radiation')
	# solar_rad_reference = lv_filter(solar_rad_reference['data'])
	# daylight_hr_mask = solar_rad_reference > 0

	#cache matrix dimensions
	# dimensions = [solar_rad_reference.shape[0], solar_rad_reference.shape[1], solar_rad_reference.shape[2]]

	#loop to extract data features
	for i, key in enumerate(filepaths):
		vars_extract[str(key)] = ncExtract(filepaths[key]) #extract files

		#break in 1-iteration to get time features & cache dimensions
		if i == 0:
			times_in = vars_extract[str(key)]['time'] 
			dimensions = [vars_extract[str(key)]['data'].shape[0], vars_extract[str(key)]['data'].shape[1], vars_extract[str(key)]['data'].shape[2]]

		vars_extract_filtered[str(key)] = lv_filter(vars_extract[str(key)]['data']) # filter data 
		# vars_extract_filtered[str(key)][~daylight_hr_mask] = 0 #mask data 
		scaler = MinMaxScaler() #normalise data
		# vars_extract_filtered_masked_norm[str(key)] = scaler.fit_transform(vars_extract_filtered[str(key)].reshape(vars_extract_filtered[str(key)].shape[0],-1)).reshape(dimensions[0], dimensions[1], dimensions[2])

	# convert u and v components to wind speed and direction
	ws = np.sqrt((vars_extract_filtered['u_wind_component']**2) + (vars_extract_filtered['v_wind_component']**2)) 
	wd = np.arctan2(vars_extract_filtered['v_wind_component'], vars_extract_filtered['u_wind_component'])
	wd = wd * (180 / np.pi)
	wd = wd + 180
	wd = 90 - wd

	# convert ws and wd to float 32
	ws = ws.astype('float32')
	wd = wd.astype('float32')

	# combine into an array
	feature_array = [ws, wd]
	# feature_array = [ws, wd, vars_extract_filtered['temperature']]

	# normalise features
	for i, array in enumerate(feature_array):
		scaler = StandardScaler(with_mean=False) #normalise data
		feature_array[i] = scaler.fit_transform(array.reshape(array.shape[0],-1)).reshape(dimensions[0], dimensions[1], dimensions[2])

	#stack features into one matrix
	# feature_array = [vars_extract_filtered_masked_norm[str(i)] for i in vars_extract_filtered_masked_norm]
	feature_array = np.stack(feature_array, axis = -1)
	# feature_array = np.concatenate((feature_array, input_timefeatures), axis = -1)

	# interpolate feature array from 24hrs to 48hrs
	feature_array = interpolate_4d(feature_array)

	# remove nan values
	outputs_mask = labels['MW'].isna().groupby(labels.index.normalize()).transform('any')

	# apply mask, removing days with more than one nan value
	feature_array = feature_array[~outputs_mask]
	labels = labels[~outputs_mask]


	#Do time feature engineering for input times
	times_in = pd.DataFrame({"datetime": times_in})
	times_in['datetime'] = times_in['datetime'].astype('str')
	times_in['datetime'] = pd.to_datetime(times_in['datetime'])
	times_in.set_index('datetime', inplace = True)
	in_times = times_in.index

	# get hours and months from datetime
	hour_in = times_in.index.hour 
	hour_in = np.float32(hour_in)

	# add HH to hours
	index = 0
	for idx, time in enumerate(hour_in):
		if time == 24:
			index += 1
		else:
			hour_in = np.insert(hour_in, index+1, time+0.5)
			index += 2

	month_in = times_in.index.month - 1 
	year_in = times_in.index.year

	# duplicate months to compensate for switch from 24hr to 48hr input data 
	index = 0
	for idx, month in enumerate(month_in):
		if idx % 24 == 0:
			index += 1
		else:
			month_in = np.insert(month_in, index+1, month)
			index += 2


	# create one_hot encoding input times: hour and month 
	one_hot_months_in = pd.get_dummies(month_in, prefix='month_')
	one_hot_hours_in = pd.get_dummies(hour_in, prefix='hour_')

	times_in_df = pd.concat([one_hot_hours_in, one_hot_months_in], axis=1)
	times_in = times_in_df.values

	# create sin / cos of input times
	times_in_hour_sin = np.expand_dims(np.sin(2*np.pi*hour_in/np.max(hour_in)), axis=-1)
	times_in_month_sin = np.expand_dims(np.sin(2*np.pi*month_in/np.max(month_in)), axis=-1)


	times_in_hour_cos = np.expand_dims(np.cos(2*np.pi*hour_in/np.max(hour_in)),axis=-1)
	times_in_month_cos = np.expand_dims(np.cos(2*np.pi*month_in/np.max(month_in)), axis=-1)


	times_in_year = (in_times - np.min(in_times)) / (np.max(in_times) - np.min(in_times))


	#Process output times as secondary input for decoder 
	#cache output times
	label_times = labels.index

	#declare 'output' time features
	df_times_outputs = pd.DataFrame()
	df_times_outputs['hour'] = labels.index.hour 
	df_times_outputs['month'] = labels.index.month - 1
	df_times_outputs['year'] = labels.index.year

	#process output times for half hours
	for idx, row in df_times_outputs.iterrows():
		if idx % 2 != 0:
			df_times_outputs.iloc[idx, 0] = df_times_outputs.iloc[idx, 0] + 0.5


	months_out = pd.get_dummies(df_times_outputs['month'], prefix='month_')
	hours_out = pd.get_dummies(df_times_outputs['hour'], prefix='hour_')

	times_out_df = pd.concat([hours_out, months_out], axis=1)
	times_out = times_out_df.values


	# create sin / cos of input times
	times_out_hour_sin = np.expand_dims(np.sin(2*np.pi*df_times_outputs['hour']/np.max(df_times_outputs['hour'])), axis=-1)
	times_out_month_sin = np.expand_dims(np.sin(2*np.pi*df_times_outputs['month']/np.max(df_times_outputs['month'])), axis=-1)

	times_out_hour_cos = np.expand_dims(np.cos(2*np.pi*df_times_outputs['hour']/np.max(df_times_outputs['hour'])), axis=-1)
	times_out_month_cos = np.expand_dims(np.cos(2*np.pi*df_times_outputs['month']/np.max(df_times_outputs['month'])), axis=-1)

	times_out_year = np.expand_dims((df_times_outputs['year'].values - np.min(df_times_outputs['year'])) / (np.max(df_times_outputs['year']) - np.min(df_times_outputs['year'])), axis=-1)

	print(times_out_hour_cos[:50])
	labels['MW'] = labels['MW'].astype('float32')

	#normalise labels
	scaler = StandardScaler(with_mean=False)
	labels[['MW']] = scaler.fit_transform(labels[['MW']])

	time_refs = [in_times, label_times]

	# one-hot method 
	# input_times = times_in_df.values
	# output_times = times_out_df.values


	# cyclic method
	# input_times = np.concatenate((times_in_hour_sin, times_in_hour_cos, times_in_month_sin, times_in_month_cos), axis=-1) swtich to output times for HH periods
	output_times = np.concatenate((times_out_hour_sin, times_out_hour_cos, times_out_month_sin, times_out_month_cos, times_out_year), axis=-1)

	labels = labels.values

	# testing input 24hr and 48hr input data - convert to 48hrs for X2
	input_times = output_times


	# add labels to inputs before 'windowing' data
	broadcaster = np.ones((feature_array.shape[0], feature_array.shape[1], feature_array.shape[2],  1), dtype=np.float32)
	broadcaster = broadcaster * np.expand_dims(np.expand_dims(labels, axis =2), axis=2)
	feature_array = np.concatenate((broadcaster, feature_array), axis = -1)

	# remove first input sequence length from output sequence
	labels = labels[input_seq_size:]
	output_times = output_times[input_seq_size:]

	# remove last output sequence from inputs
	input_times = input_times[:-output_seq_size]
	feature_array = feature_array[:-output_seq_size]

	# remove deleted times from time refs
	in_times = label_times[:-output_seq_size]
	label_times = label_times[input_seq_size:]

	# split train and test data
	num_seq = len(labels) - (output_seq_size - 1)
	test_set_percentage = 0.1


	test_split_seq = int(np.floor(num_seq * (1 - test_set_percentage)))
	print(test_split_seq)
	

	input_test_seq =  test_split_seq + (input_seq_size - 1)
	output_test_seq = test_split_seq + (output_seq_size - 1)

	# create dataset
	dataset = {
		'train_set' : {
			'X1_train': feature_array[:input_test_seq],
			'X2_train': input_times[:input_test_seq], # input time features
			'X3_train': output_times[:output_test_seq], # output time features
			'y_train': labels[:output_test_seq] 
			},
		'test_set' : {
			'X1_test': feature_array[input_test_seq:],
			'X2_test': input_times[input_test_seq:], 
			'X3_test': output_times[output_test_seq:],
			'y_test': labels[output_test_seq:] 
			}
		}

	time_refs = {
		'input_times_train': in_times[:input_test_seq],
		'input_times_test': in_times[input_test_seq:], 
		'output_times_train': label_times[:output_test_seq],
		'output_times_test': label_times[output_test_seq:]
	}

	# print(dataset['train_set']['X1_train'].shape)
	# print(dataset['test_set']['X1_test'].shape)
	# exit()

	# print(feature_array.shape)
	# print(labels.shape)
	# print(f'input: {in_times[-1]}')
	# print(f'output: {label_times[-1]}')

	# a = pd.DataFrame(time_refs[1])
	# a = a.set_index('datetime').resample('1h')
	# a.to_clipboard()
	# exit()


	#divide into timesteps & train and test sets
	# dataset, time_refs = format_data_into_timesteps(X1 = feature_array, X2 = input_times , X3 = output_times, Y = labels, input_seq_size = 240, output_seq_size = 48, input_times_reference = time_refs[1], output_times_reference = time_refs[1]) # converting from 24hr to 48hr inputs hence can use output time references
	# train_set, test_set, time_refs

	# def to_float32(input_dict):
	# 	for idx, key in enumerate(input_dict.keys()):
	# 		input_dict[key] = input_dict[key].astype(np.float32)
	# 	return input_dict

	# train_set = to_float32(train_set)
	# test_set = to_float32(test_set)	

	return dataset, time_refs
	# return train_set, test_set, time_refs



#paths to nc files for x_value features:
filepaths = {
	 'u_wind_component': './Data/wind/Raw_Data/10m_u_component_of_wind',
	 'v_wind_component': './Data/wind/Raw_Data/10m_v_component_of_wind',
	 # 'temperature': './Data/wind/Raw_Data/temperature_data',
}

#load labels (solar generation per HH)
windGenLabels = pd.read_csv('./Data/wind/Raw_Data/HH_windGen_V2.csv', parse_dates=True, index_col=0, header=0, dayfirst=True)



dataset, time_refs = data_processing(filepaths = filepaths, labels = windGenLabels, input_seq_size = 240, output_seq_size = 48)
# train_set, test_set, time_refs 

# print(*[f'{key}: {train_set[key].shape}' for key in train_set.keys()], sep='\n')
# print(*[f'{key}: {test_set[key].shape}' for key in test_set.keys()], sep='\n')

# #save training set as dictionary
# with open("./Data/wind/Processed_Data/train_set_V1_withtimefeatures_96hrinput_24hrs.pkl", "wb") as trainset:
# 	dump(train_set, trainset)

# #save training set as dictionary
# with open("./Data/wind/Processed_Data/test_set_V1_withtimefeatures_96hrinput_24hrs.pkl", "wb") as testset:
# 	dump(test_set, testset)		
	
# #save time timeseries (inputs & outputs) for reference
with open("./Data/wind/Processed_Data/time_refs_V5_withtimefeatures_120hrinput.pkl", "wb") as times:
	dump(time_refs, times)

# save training set as dictionary (h5py dump)
f = h5py.File('./Data/wind/Processed_Data/train_set_V5_withtimefeatures_120hrinput_float32.hdf5', 'w')

for group_name in dataset:
	group = f.create_group(group_name)
	for dset_name in dataset[group_name]:
		dset = group.create_dataset(dset_name, data = dataset[group_name][dset_name])
f.close()

exit()


###########################################_____LOAD_PROCESSED_DATA_____#############################################

#load training data dictionary
train_set_load = open("./Data/solar/Processed_Data/train_setv5_float32.pkl", "rb") 
train_set = load(train_set_load)
train_set_load.close()


#load times references
time_set_load = open("./Data/solar/Processed_Data/time_refsv5.pkl", "rb") 
time_refs = load(time_set_load)
time_set_load.close()

print(train_set['X1_train'].shape)
exit()



###########################################_____DATA_GENERATOR_____#################################################

params = {'dim': (24, 46, 55),
		'batch_size': 32,
		'n_channels': 8,
		'inputSize': 24,
		'outputSize': 48 } #sequence lengths relevant to python indexes

class DataGenerator(keras.utils.Sequence):

	def __init__(self, features, output_datetime_features, labels, batch_size, dim, n_channels, inputSize, outputSize):
		self.dim = dim
		self.features = features
		self.batch_size = batch_size
		self.labels = labels
		self.n_channels = n_channels
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.output_datetime_features = output_datetime_features
		self.on_epoch_end()

	def __len__(self):
		# 'number of batches per Epoch'
		return int(np.floor(len(self.features)/ self.batch_size))

	def __getitem__(self, index):

		# Generate data
		X_train1, X_train2, y_train = self.__data_generation(index)

		return X_train1, X_train2, y_train

	def on_epoch_end(self):
		# set length of indexes for each epoch
		self.input_indexes = np.arange(len(self.features))
		self.output_indexes = np.arange(len(self.labels))


	def __data_generation(self, index):
		# Generate training data
		X_train1 = self.features[(index*self.batch_size):((index+1)*self.batch_size)]
		X_train2 = self.output_datetime_features[(index*self.batch_size):((index+1)*self.batch_size)]

		y_train = self.labels[(index*self.batch_size):((index+1)*self.batch_size)]

		return X_train1, X_train2, y_train

training_generator = DataGenerator(features = train_set['X1_train'], output_datetime_features = train_set['X2_train'], labels = train_set['y_train'], **params)


###########################################_____MODEL_SETUP_____##############################################


#######___NEW_MODEL___############

class Encoder(Model):

	def __init__(self, batch_size):
		super(Encoder, self).__init__()

		self.batch_size = batch_size
		# self.input_shape = input_shape

		self.ConvLSTM2D_1 = ConvLSTM2D(filters=32, 
			kernel_size=(3, 3), 
			data_format='channels_last', 
			recurrent_activation='relu', 
			activation='tanh', 
			padding='same', 
			return_sequences=True)

		self.ConvLSTM2D_2 = ConvLSTM2D(filters=32, 
			kernel_size=(3, 3), 
			data_format='channels_last', 
			recurrent_activation='relu', 
			activation='tanh', 
			padding='same', 
			return_sequences=True)

		self.ConvLSTM2D_3 = ConvLSTM2D(filters=12, 
			kernel_size=(3, 3), 
			data_format='channels_last', 
			recurrent_activation='relu', 
			activation='tanh', 
			padding='same', 
			return_sequences=True)

		self.maxPool_1 = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')
		self.maxPool_2 = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')



	def call(self, input):

		convLSTM_1 = Bidirectional(self.ConvLSTM2D_1(input))
		batchNormalisation_1 = BatchNormalization(convLSTM_1)
		maxPooling_1 = self.maxPool_1(batchNormalisation_1)

		convLSTM_2 = Bidirectional(self.ConvLSTM2D_2(maxPooling_1))
		batchNormalisation_2 = BatchNormalization(convLSTM_2)
		maxPooling_2 = self.maxPool_2(batchNormalisation_2)

		encoder_outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(self.ConvLSTM2D_3(maxPooling_2)) 


		return encoder_outputs, forward_h, forward_c, backward_h, backward_c

	def inital_hidden_states(self):
		forward_h = tf.zeros((self.batch_size, 12, 14, 12))
		forward_c = tf.zeros((self.batch_size, 12, 14, 12))
		backward_h = tf.zeros((self.batch_size, 12, 14, 12))
		backward_c = tf.zeros((self.batch_size, 12, 14, 12))


class Decoder(Model):

	def __init__(self, batch_size):
		super(Decoder, self).__init__()

		self.batch_size = batch_size
		# self.output_shape = output_shape
		self.conv1d_1 = Conv1D(32, kernel_size=1, strides=1, padding='same', data_format='channels_last', activation='relu')
		self.conv1d_2 = Conv1D(32, kernel_size=1, strides=1, padding='same', data_format='channels_last', activation='relu')
		# self.conv2d = Conv2D(12, kernel_size=(3,3), stides=1, padding='same', data_format='channels_last', activation='softmax')
		# self.tanh = keras.Activation('tanh')
		self.conv1d_v = Conv1D(1, kernel_size=1, strides=1, padding='same', data_format='channels_last', activation='sigmoid')

		self.lstm = LSTM(512, return_sequences=True, return_state=True)
		self.conv1d_output = Conv1D(1, kernel_size=1, strides=1, padding='same', activation='linear')


	def call(self, decoder_input, output, forward_h, forward_c, backward_h, backward_c):

		#concat forward & back hidden states
		state_h_comb = concatenate([forward_h, backward_h])
		state_c_comb = concatenate([forward_c, backward_c])

		#encoder hidden layer outputs
		decoder_hidden = [state_h_comb, state_c_comb]

		#Exapnd for time
		state_h_time = keras.expand_dims(state_h_comb, axis = 1)
		state_c_time = keras.expand_dims(state_c_comb, axis = 1)
		state_hc_time = concatenate([forward_c, backward_c], axis = -1)

		#calculate addidative score
		score_s = self.conv1d_1(encoder_output)
		score_h = self.conv1d_2(state_hc_time)
		add_score = self.tanh(score_s + score_h)
		add_score = self.conv1d_v(add_score)

		#get attention weights
		attention_weights = softmax(add_score, axis=1)

		#context_vector
		context_vector = attention_weights * encoder_output
		context_vector = keras.reduce_sum(context_vector, axis = 1)
		context_vector = keras.expand_dims(context_vector, axis = 1)
		context_vector = concatenate([context_vector, decoder_input], axis=-1)

		decoder_output, state_h, state_c = self.lstm(context_vector, initial_state= decoder_hidden)

		output = self.conv1d_output(decoder_output)


		return context_vector, attention_weights

Encoder = Encoder(32)
Decoder = Decoder(32)

# class Decoder(Model):

# 	def __init__(self, decoder_input, decoder_hidden, encoder_output):
# 		super(Decoder, self).__init__()

# 		self.output_shape = output_shape




# 	def call(self, ):

# 		lstm_layer_1 = Bidirectional(LSTM(context_vector, activation = 'relu', return_sequences = True, initial_state = encoder_states))
# 		batchNormalisation_6 = BatchNormalization()(lstm_layer_1)
# 		dense_1 = TimeDistributed(Dense(64, activation = 'relu'))(batchNormalisation_7)
# 		outputs = TimeDistributed(Dense(1))(dense_1)


# 		return predictions, decoder_hidden, attention_weights



#define loss function
def pinball_loss(q,y,f):
	e = (y - f)
	return K.mean(K.maximum(q*e, (q-1) * e), axis = -1)

#define optimiser 
optimizer = keras.optimizers.Adam()


#prediction / training setup 
def train_step(input, target):
	loss = 0

	with tf.GradientTape() as tape:
		encoder_outputs, forward_h, forward_c, backward_h, backward_c = Encoder(input)

		decoder_hidden = [forward_h, forward_c, backward_h, backward_c]

		decoder_input = input[:,-1:,:,:,:]

		#Teacher Forcing - feeding the target as the next input
		for t in range(0, target.shape[1]):

			#pass encoder to the decoder
			predictions, decoder_hidden, attention_weights = Decoder(decoder_input, decoder_hidden, encoder_output)

			loss += pinball_loss(target[:, t:t+1, 0:1], predictions)

			decoder_input = predictions, target[:, t:t+1, :]

	train_loss +=  (loss / int(target.shape[1]))

	#training with the loss
	varaibles = Encoder.trainable_variables + Decoder.trainable_variables
	gradients = tape.gradient(loss, varaibles)
	optmisier.apply_gradients(zip(gradients, variables))

	return batch_loss


#Define the number EPOCHS
EPOCHS = 10

#training protocol
for epoch in range(EPOCHS):
	start = time.time()

	encoder_hidden = Encoder.inital_hidden_states()
	total_loss = 0

	for batch, (X_train1, X_train2, y_train) in enumerate(training_generator):
		batch_loss = train_step(X_train1, y_train)
		total_loss += batch_loss

		# if batch % 10 == 0:
		print('Epoch {} Batch {} Loss {:.4f}' .format(epoch + 1, batch, batch_loss.numpy()))


	print('Epoch {} Loss {:.4f}' .format(epoch + 1, total_loss / steps_per_epoch))
	print('Time taken for Epoch {} sec\n' .format(time.time() - start))


##################################





# def solarGenation_Model():

# 	#get dimensions for inputs
# 	samples, timestep, rows, cols, channels= (train_set['X1_train'].shape[i] for i in range(5))
# 	time_features = train_set['X2_train'].shape[-1]
# 	output_seq_size = 48 #output 

# 	input_layer1 = Input(shape=(timestep, rows, cols, channels))

# 	#Define the the ConvLSTM encoder for 2D Data
# 	convLSTM_1 = Bidirectional(ConvLSTM2D(filters=32, 
# 		kernel_size=(3, 3), 
# 		data_format='channels_last', 
# 		recurrent_activation='relu', 
# 		activation='tanh', 
# 		padding='same', 
# 		return_sequences=True))(input_layer1)
# 	batchNormalisation_1 = BatchNormalization()(convLSTM_1)
# 	maxPooling_1 = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')(batchNormalisation_1)

# 	convLSTM_2 = Bidirectional(ConvLSTM2D(filters=32, 
# 		kernel_size=(3, 3), 
# 		data_format='channels_last', 
# 		recurrent_activation='relu', 
# 		activation='tanh', 
# 		padding='same', 
# 		return_sequences=True))(maxPooling_1)
# 	batchNormalisation_2 = BatchNormalization()(convLSTM_2)
# 	maxPooling_2 = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')(batchNormalisation_2)

# 	# convLSTM_3 = ConvLSTM2D(filters=16, 
# 	# 	kernel_size=(3, 3), 
# 	# 	data_format='channels_last',  
# 	# 	recurrent_activation='relu',
# 	# 	activation='tanh', 
# 	# 	padding='same', 
# 	# 	return_sequences=True)(maxPooling_2)
# 	# batchNormalisation_3 = BatchNormalization()(convLSTM_3)
# 	# maxPooling_3 = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')(batchNormalisation_3)

# 	# convLSTM_4 = ConvLSTM2D(filters=16, 
# 	# 	kernel_size=(5, 5), 
# 	# 	data_format='channels_last',  
# 	# 	recurrent_activation='relu',
# 	# 	activation='tanh',  
# 	# 	padding='same', 
# 	# 	return_sequences=True)(maxPooling_3)
# 	# batchNormalisation_4 = BatchNormalization()(convLSTM_4)
# 	# maxPooling_4 = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')(batchNormalisation_4)

# 	# convLSTM_5 = ConvLSTM2D(filters=16, 
# 	# 	kernel_size=(2, 2), 
# 	# 	# data_format='channels_last',  
# 	# 	recurrent_activation='relu',
# 	# 	activation='tanh', 
# 	# 	padding='same', 
# 	# 	return_sequences=True)(maxPooling_4)
# 	# batchNormalisation_5 = BatchNormalization()(convLSTM_5)
# 	# maxPooling_5 = MaxPooling3D(pool_size=(1, 1, 1), padding='same', data_format='channels_last')(batchNormalisation_5)

# 	encoder_outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(ConvLSTM2D(filters=12, 
# 		kernel_size=(3, 3), 
# 		data_format='channels_last', 
# 		stateful = False, 
# 		kernel_initializer = 'random_uniform',
# 		padding='same', 
# 		return_sequences=True,
# 		return_state=True))(maxPooling_2)
# 	# batchNormalisation_6 = BatchNormalization()(convLSTM_6)
# 	# maxPooling_6 = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')(batchNormalisation_6)

# 	# flat_layer = Flatten()(maxPooling_6)
# 	# repeatVector1 = RepeatVector(output_seq_size)(flat_layer)

# 	#Define MLP encoder for input time features
# 	# input_layer2 = Input(shape=(timestep, time_features))
# 	# denseLayer1 = Dense(8, activation='relu')(input_layer2)
# 	# flat_layer2 = Flatten()(denseLayer1)
# 	# repeatVector2 = RepeatVector(output_seq_size)(flat_layer2)

# 	#bring in output time features
# 	input_layer2 = Input(shape=(output_seq_size, time_features))

# 	print(forward_h.shape)
# 	print(forward_c.shape)

# 	flat_layer_fc = Flatten()(forward_c)
# 	# repeatVector1 = RepeatVector(output_seq_size)(flat_layer_statec)

# 	flat_layer_fh = Flatten()(forward_h)
# 	# repeatVector2 = RepeatVector(output_seq_size)(flat_layer_stateh)

# 	flat_layer_bc = Flatten()(backward_c)
# 	# repeatVector1 = RepeatVector(output_seq_size)(flat_layer_statec)

# 	flat_layer_bh = Flatten()(backward_h)

    
# 	print(flat_layer_fc.shape)

# 	encoder_states = [flat_layer_fh, flat_layer_fc, flat_layer_bh, flat_layer_bc]



# 	# concat = concatenate([repeatVector1, input_layer2])

# 	# denseLayer2 = Dense(128, activation='relu')(concat_Inputs)
# 	# denseLayer3 = Dense(32, activation='relu')(denseLayer2) 
# 	# batchNormalisation_5 = BatchNormalization()(denseLayer3)
# 	# flat_layer3 = Flatten()(batchNormalisation_5)
# 	# repeatVector3 = RepeatVector(output_seq_size)(flat_layer3)

# 	#Define decoder LSTM
# 	lstm_layer_1 = Bidirectional(LSTM(flat_layer_fh.shape[1], activation = 'relu', return_sequences = True))(input_layer2, initial_state = encoder_states)
# 	batchNormalisation_6 = BatchNormalization()(lstm_layer_1)
# 	lstm_layer_2 = Bidirectional(LSTM(128, activation = 'relu', return_sequences = True))(batchNormalisation_6)
# 	batchNormalisation_7 = BatchNormalization()(lstm_layer_2)
# 	# lstm_layer_3 = Bidirectional(LSTM(128, activation = 'relu', return_sequences = True))(batchNormalisation_7)
# 	# batchNormalisation_8 = BatchNormalization()(lstm_layer_3)
# 	dense_1 = TimeDistributed(Dense(64, activation = 'relu'))(batchNormalisation_7)
# 	# dense_2 = TimeDistributed(Dense(32, activation = 'relu'))(dense_1)

# 	outputs = TimeDistributed(Dense(1))(dense_1)

# 	model = Model(inputs=[input_layer1, input_layer2], outputs=outputs)

# 	return model

# #define the pinball loss function to optimise
# def defined_loss(q,y,f):
# 	e = (y - f)
# 	return K.mean(K.maximum(q*e, (q-1) * e), axis = -1)



# #declare quantiles
# quantiles = ['0.' + str(i) for i in range(1,10)]
# quantiles = list(map(float, quantiles))
# quantiles.append(0.99)
# quantiles.insert(0, 0.01)

# print(quantiles)

# quantiles = [0.5]
# # quantiles = ['0.' + str(i) for i in range(1,3)] #for testing
# quantiles = list(map(float, quantiles)) #for testing

# print(quantiles)

# #include clipvalue in optmisier
# optimizer = keras.optimizers.Adam()

# yhats = {}

# #train each model for each quantile
# for q in quantiles:
# 	print(q)
# 	model = solarGenation_Model()
# 	model.compile(loss = lambda y,f: defined_loss(q,y,f), optimizer= optimizer, metrics = ['mae'])
# 	print(model.summary())
# 	train = model.fit(training_generator, epochs = 1, shuffle = False, use_multiprocessing = False)
# 	# yhats[str(q)] = model.predict(X_test)
# 	model.save('solarGeneration_forecast_increasedfiltersTEST' + '_Q_%s' %(q) + '.h5')
# 	K.clear_session()

# #visulise training
# plt.plot(train.history['loss'])
# plt.xlabel('epoch')
# plt.legend()
# plt.show()
 

# predictions = {}
# for i, key in enumerate(yhats.keys()):
# 	plt.plot(yhats[key][5,:,:], label=key)

# plt.legend()
# plt.show()

