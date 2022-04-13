import pandas as pd
import numpy as np
import sys
import os
from pickle import dump, load
import h5py

from preprocessing_funcs import demand_data_processing

np.set_printoptions(threshold=sys.maxsize)

###########################################_____LOAD & PRE-PROCESS DATA_____###########################################

#cache current working directory of main script
workingDir = os.getcwd()

# paths to nc files for x_value features:
filepaths = {
	 'temperature': '../../data/raw/temperature',
}

#load labels (solar generation per HH)
demandGenLabels = pd.read_csv('../../data/raw/demand_labels/HH_demand.csv', parse_dates=True, index_col=0, header=0, dayfirst=True)

# call main pre-processing function - sequence windowing no longer utilised
dataset, time_refs = demand_data_processing(filepaths = filepaths, labels = demandGenLabels, workingDir = workingDir)

# print data summaries
print(*[f'{key}: {dataset["train_set"][key].shape}' for key in dataset['train_set'].keys()], sep='\n')
print(*[f'{key}: {dataset["test_set"][key].shape}' for key in dataset['test_set'].keys()], sep='\n')

# #save time timeseries (inputs & outputs) for reference
print('saving data...')
with open("../../data/processed/demand/time_refs_demand.pkl", "wb") as times:
	dump(time_refs, times)

# save training set as dictionary (h5py dump)
f = h5py.File('../../data/processed/demand/dataset_demand.hdf5', 'w')

for group_name in dataset:
	group = f.create_group(group_name)
	for dset_name in dataset[group_name]:
		dset = group.create_dataset(dset_name, data = dataset[group_name][dset_name])
f.close()

