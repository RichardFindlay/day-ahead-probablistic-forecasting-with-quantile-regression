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
from workalendar.europe import UnitedKingdom
cal = UnitedKingdom()

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
# def interpolate_time(time_array):
# 	interp_time = np.linspace(time_array[0], time_array[-1], len(time_array)*2)

# 	return interp_time




def format_data_into_timesteps(X1, X2, X3, Y, input_seq_size, output_seq_size, input_times_reference, output_times_reference):
	print('formating data into timesteps & interpolating input data')

	#number of timesteps to be included in each sequence
	seqX1, seqX2, seqX3, seqY_in, seqY, in_times, out_times = [], [], [], [], [], [], []
	input_start, input_end = 0, 0
	output_start = input_seq_size - output_seq_size 
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
		

		input_start += output_seq_size  # divide by 2 to compensate for 24hr period (edited)
		output_start += output_seq_size
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
def data_processing(filepaths, labels):

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
		vars_extract_filtered_masked_norm[str(key)] = scaler.fit_transform(vars_extract_filtered[str(key)].reshape(vars_extract_filtered[str(key)].shape[0],-1)).reshape(dimensions[0], dimensions[1], dimensions[2])


	#stack features into one matrix
	feature_array = [vars_extract_filtered_masked_norm[str(i)] for i in vars_extract_filtered_masked_norm]
	feature_array = np.stack(feature_array, axis = -1)
	# feature_array = np.concatenate((feature_array, input_timefeatures), axis = -1)

	# interpolate feature array from 24hrs to 48hrs
	feature_array = interpolate_4d(feature_array)

	
	# remove nan values
	outputs_mask = labels['quantity (MW)'].isna().groupby(labels.index.normalize()).transform('any')

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
	df_times_outputs['date'] = labels.index.date
	df_times_outputs['hour'] = labels.index.hour 
	df_times_outputs['month'] = labels.index.month - 1
	df_times_outputs['year'] = labels.index.year
	df_times_outputs['day_of_week'] = labels.index.dayofweek
	df_times_outputs['day_of_year'] = labels.index.dayofyear - 1
	df_times_outputs['weekend'] = df_times_outputs['day_of_week'].apply(lambda x: 1 if x>=5 else 0)


	# account for bank / public holidays
	start_date = labels.index.min()
	end_date = labels.index.max()
	start_year = df_times_outputs['year'].min()
	end_year = df_times_outputs['year'].max()

	
	holidays = set(holiday[0] 
		for year in range(start_year, end_year + 1) 
		for holiday in cal.holidays(year)
		if start_date <=  holiday[0] <= end_date)

	df_times_outputs['holiday'] = df_times_outputs['date'].isin(holidays).astype(int)

	#process output times for half hours
	for idx, row in df_times_outputs.iterrows():
		if idx % 2 != 0:
			df_times_outputs.iloc[idx, 1] = df_times_outputs.iloc[idx, 1] + 0.5


	months_out = pd.get_dummies(df_times_outputs['month'], prefix='month_')
	hours_out = pd.get_dummies(df_times_outputs['hour'], prefix='hour_')

	times_out_df = pd.concat([hours_out, months_out], axis=1)
	times_out = times_out_df.values


	# create sin / cos of output hour
	times_out_hour_sin = np.expand_dims(np.sin(2*np.pi*df_times_outputs['hour']/np.max(df_times_outputs['hour'])), axis=-1)
	times_out_hour_cos = np.expand_dims(np.cos(2*np.pi*df_times_outputs['hour']/np.max(df_times_outputs['hour'])), axis=-1)

	# create sin / cos of output month
	times_out_month_sin = np.expand_dims(np.sin(2*np.pi*df_times_outputs['month']/np.max(df_times_outputs['month'])), axis=-1)
	times_out_month_cos = np.expand_dims(np.cos(2*np.pi*df_times_outputs['month']/np.max(df_times_outputs['month'])), axis=-1)

	# create sin / cos of output year
	# times_out_year = np.expand_dims((df_times_outputs['year'].values - np.min(df_times_outputs['year'])) / (np.max(df_times_outputs['year']) - np.min(df_times_outputs['year'])), axis=-1)

	# create sin / cos of output day of week
	times_out_DoW_sin = np.expand_dims(np.sin(2*np.pi*df_times_outputs['day_of_week']/np.max(df_times_outputs['day_of_week'])), axis=-1)
	times_out_DoW_cos = np.expand_dims(np.cos(2*np.pi*df_times_outputs['day_of_week']/np.max(df_times_outputs['day_of_week'])), axis=-1)

	# create sin / cos of output day of year
	times_out_DoY_sin = np.expand_dims(np.sin(2*np.pi*df_times_outputs['day_of_year']/np.max(df_times_outputs['day_of_year'])), axis=-1)
	times_out_DoY_cos = np.expand_dims(np.cos(2*np.pi*df_times_outputs['day_of_year']/np.max(df_times_outputs['day_of_year'])), axis=-1)		


	print(times_out_hour_cos[:50])
	#normalise labels
	scaler = StandardScaler(with_mean=False)
	labels[['quantity (MW)']] = scaler.fit_transform(labels[['quantity (MW)']])

	time_refs = [in_times, label_times]

	# one-hot method 
	# input_times = times_in_df.values
	# output_times = times_out_df.values

	weekends = np.expand_dims(df_times_outputs['weekend'].values, axis =-1)
	holidays = np.expand_dims(df_times_outputs['holiday'].values, axis =-1)



	# cyclic method
	# input_times = np.concatenate((times_in_hour_sin, times_in_hour_cos, times_in_month_sin, times_in_month_cos), axis=-1) swtich to output times for HH periods
	output_times = np.concatenate((times_out_hour_sin, times_out_hour_cos, times_out_month_sin, times_out_month_cos, times_out_DoW_sin, times_out_DoW_cos,
									 times_out_DoY_sin, times_out_DoY_cos, weekends, holidays), axis=-1)


	labels = labels.values

	# testing input 24hr and 48hr input data - convert to 48hrs for X2
	input_times = output_times

	print(feature_array.shape)
	print(labels.shape)
	print(f'input: {time_refs[0][-1]}')
	print(f'output: {time_refs[1][-1]}')

	# a = pd.DataFrame(time_refs[1])
	# a = a.set_index('datetime').resample('1h')
	# a.to_clipboard()
	# exit()

	# add labels to inputs
	print('combining feature array with lagged outputs')
	broadcaster = np.ones((feature_array.shape[0], feature_array.shape[1], feature_array.shape[2],  1), dtype=np.float32)
	broadcaster = broadcaster * np.expand_dims(np.expand_dims(labels, axis =2), axis=2)
	feature_array = np.concatenate((broadcaster, feature_array), axis = -1)


	#divide into timesteps & train and test sets
	# dataset, time_refs = format_data_into_timesteps(X1 = feature_array, X2 = input_times , X3 = output_times, Y = labels, input_seq_size = 240, output_seq_size = 48, input_times_reference = time_refs[1], output_times_reference = time_refs[1]) # converting from 24hr to 48hr inputs hence can use output time references
	# train_set, test_set, time_refs

	# def to_float32(input_dict):
	# 	for idx, key in enumerate(input_dict.keys()):
	# 		input_dict[key] = input_dict[key].astype(np.float32)
	# 	return input_dict

	# train_set = to_float32(train_set)
	# test_set = to_float32(test_set)	

	test_split_seq = 4800 # use the last 100 days, around 10%
	
	# input_test_seq =  test_split_seq + (input_seq_size - 1)
	# output_test_seq = test_split_seq + (output_seq_size - 1)




	#divide into timesteps & train and test sets
	# dataset, time_refs = format_data_into_timesteps(X1 = feature_array, X2 = input_times , X3 = output_times, Y = labels, input_seq_size = 240, output_seq_size = 48, input_times_reference = time_refs[1], output_times_reference = time_refs[1]) # converting from 24hr to 48hr inputs hence can use output time references

	# create dataset
	dataset = {
		'train_set' : {
			'X1_train': feature_array[:-test_split_seq],
			'X2_train': input_times[:-test_split_seq], # input time features
			'X3_train': output_times[:-test_split_seq], # output time features
			'y_train': labels[:-test_split_seq] 
			},
		'test_set' : {
			'X1_test': feature_array[-test_split_seq:],
			'X2_test': input_times[-test_split_seq:], 
			'X3_test': output_times[-test_split_seq:],
			'y_test': labels[-test_split_seq:] 
			}
		}

	time_refs = {
		'input_times_train': label_times[:-test_split_seq],
		'input_times_test': label_times[-test_split_seq:], 
		'output_times_train': label_times[:-test_split_seq],
		'output_times_test': label_times[-test_split_seq:]
	}



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
	 # 'u_wind_component': './Data/wind/Raw_Data/10m_u_component_of_wind',
	 # 'v_wind_component': './Data/wind/Raw_Data/10m_v_component_of_wind',
	 'temperature': './Data/wind/Raw_Data/temperature_data',
}

#load labels (solar generation per HH)
windGenLabels = pd.read_csv('./Data/demand/Raw_Data/uk_demand_20190101_20210630_V2.csv', parse_dates=True, index_col=0, header=0, dayfirst=True)

dataset, time_refs = data_processing(filepaths = filepaths, labels = windGenLabels)
# train_set, test_set, time_refs 

# #save training set as dictionary
# with open("./Data/wind/Processed_Data/train_set_V1_withtimefeatures_96hrinput_24hrs.pkl", "wb") as trainset:
# 	dump(train_set, trainset)
print(*[f'{key}: {dataset["train_set"][key].shape}' for key in dataset['train_set'].keys()], sep='\n')
print(*[f'{key}: {dataset["test_set"][key].shape}' for key in dataset['test_set'].keys()], sep='\n')


# #save training set as dictionary
# with open("./Data/wind/Processed_Data/test_set_V1_withtimefeatures_96hrinput_24hrs.pkl", "wb") as testset:
# 	dump(test_set, testset)		
	
# #save time timeseries (inputs & outputs) for reference
with open("./Data/demand/Processed_Data/time_refs_V2_withtimefeatures_Demand.pkl", "wb") as times:
	dump(time_refs, times)

# save training set as dictionary (h5py dump)
f = h5py.File('./Data/demand/Processed_Data/dataset_V2_withtimefeatures_Demand.hdf5', 'w')

for group_name in dataset:
	group = f.create_group(group_name)
	for dset_name in dataset[group_name]:
		dset = group.create_dataset(dset_name, data = dataset[group_name][dset_name])
f.close()

exit()


###########################################_____LOAD_PROCESSED_DATA_____#############################################
