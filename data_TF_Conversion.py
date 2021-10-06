import numpy as np
from pickle import load, dump

import matplotlib.pyplot as plt
import h5py
import os


# convert Xtrain3, y_train and time_refs from [len, 48, 1] to [len*48, 1, 1]




time_set_load = open("./Data/wind/Processed_Data/time_refs_V2_withtimefeatures_96hrinput.pkl", "rb") 
time_set = load(time_set_load)
time_set_load.close()


f = h5py.File('./Data/wind/Processed_Data/train_set_V2_withtimefeatures_96hrinput_float32.hdf5', 'r')


print(f['test_set']['y_test'].shape)
print(f['test_set']['X3_test'].shape)

exit()


# update x3_train
old_x3_train = f['train_set']['X3_train']
old_x3_train = np.array(old_x3_train)
new_x3_train = old_x3_train.reshape(-1, old_x3_train.shape[-1])

# update y_true train
old_ytrue_train = f['train_set']['y_train']
old_ytrue_train  = np.array(old_ytrue_train)
new_ytrue_train = old_ytrue_train.reshape(-1, old_ytrue_train.shape[-1])


# update x3_test
old_x3_test = f['test_set']['X3_test']
old_x3_test = np.array(old_x3_test)
new_x3_test = old_x3_test.reshape(-1, old_x3_test.shape[-1])


# update y_true train
old_ytrue_test = f['test_set']['y_test']
old_ytrue_test  = np.array(old_ytrue_test)
new_ytrue_test = old_ytrue_test.reshape(-1, old_ytrue_test.shape[-1])





f.close()
# plt.plot(old_ytrue_train[48, :, 0])
# plt.plot(new_ytrue_train[2304:2352, 0])
# plt.show()



# print(time_set['output_times_train'][10])


# save training set as dictionary (h5py dump)
data = h5py.File('./Data/wind/Processed_Data/train_set_V2_withtimefeatures_96hrinput_float32.hdf5', 'r+')

# # deleted and assign new vars
# del data['train_set']['X3_train']
# data['train_set']['X3_train'] = new_x3_train

# # deleted and assign new vars
# del data['train_set']['y_train']
# data['train_set']['y_train'] = new_ytrue_train



# deleted and assign new vars
# del data['test_set']['X3_test']
# data['test_set']['X3_test'] = new_x3_test

# # deleted and assign new vars
# del data['test_set']['y_test']
# data['test_set']['y_test'] = new_ytrue_test




data.close()



