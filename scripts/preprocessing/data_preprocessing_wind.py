import pandas as pd
import numpy as np
import sys
import os
from pickle import dump, load
import h5py

from preprocessing_funcs import wind_data_processing


np.set_printoptions(threshold=sys.maxsize)

###########################################_____LOAD & PRE-PROCESS DATA_____###########################################

#cache current working directory of main script
workingDir = os.getcwd()

# paths to nc files for x_value features:
filepaths = {
	 'u_wind_component_10': '../../data/raw/10m_u_component_of_wind',
	 'v_wind_component_10': '../../data/raw/10m_v_component_of_wind',
 	 'u_wind_component_100': '../../data/raw/100m_u_component_of_wind',
	 'v_wind_component_100': '../../data/raw/100m_v_component_of_wind',
	 'instantaneous_10m_wind_gust': '../../data/raw/instantaneous_10m_wind_gust',
	 'surface_pressure': '../../data/raw/surface_pressure',
	 'temperature': '../../data/raw/temperature'
}

#load labels (wind generation per HH)
windGenLabels = pd.read_csv('../../data/raw/wind_labels/HH_windGen_v4.csv', parse_dates=True, index_col=0, header=0, dayfirst=True)

# call main pre-processing function - sequence windowing no longer utilised
dataset, time_refs = wind_data_processing(filepaths = filepaths, labels = windGenLabels, input_seq_size = 336, output_seq_size = 48, workingDir = workingDir)

# print data summaries
print(*[f'{key}: {dataset["train_set"][key].shape}' for key in dataset['train_set'].keys()], sep='\n')
print(*[f'{key}: {dataset["test_set"][key].shape}' for key in dataset['test_set'].keys()], sep='\n')

# #save time timeseries references (inputs & outputs) for reference
print('saving data...')
with open("../../data/processed/wind/time_refs_wind_v4.pkl", "wb") as times:
	dump(time_refs, times)

# save training set as dictionary (h5py dump)
f = h5py.File('../../data/processed/wind/dataset_wind_v4.hdf5', 'w')

for group_name in dataset:
	group = f.create_group(group_name)
	for dset_name in dataset[group_name]:
		dset = group.create_dataset(dset_name, data = dataset[group_name][dset_name])
f.close()





