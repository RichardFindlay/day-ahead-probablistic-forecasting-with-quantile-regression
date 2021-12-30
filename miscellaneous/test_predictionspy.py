

import numpy as np
from pickle import load
import matplotlib.pyplot as plt
from keras import Model
from keras.models import load_model, model_from_json

from attentionlayer import attention
import sys
import h5py






# type ="wind"

# if type == 'wind':
# 	dataset_name = 'train_set_V100.hdf5'
# elif type == 'demand':
# 	dataset_name = 'dataset_V1_withtimefeatures_Demand.hdf5'
# elif type == 'solar':
# 	dataset_name = 'train_set_V21_withtimefeatures_120hrinput.hdf5'

# # dataset_name = 'train_set_V6_withtimefeatures_120hrinput_float32.hdf5'
# f = h5py.File(f"./Data/{type}/Processed_Data/{dataset_name}", "r")
# features = np.empty_like(f['train_set']['X1_train'][0])
# times_in = np.empty_like(f['train_set']['X2_train'][0])
# times_out = np.empty_like(f['train_set']['X3_train'][0])
# labels = np.empty_like(f['train_set']['y_train'][0])
# x_len = f['train_set']['X1_train'].shape[0]
# y_len = f['train_set']['y_train'].shape[0]
# print('size parameters loaded')
# f.close()  

# input_seq_size = 672
# output_seq_size = 1
# n_s = 128
# Q = 0.1

# # make custom activation - swish
# from keras.backend import sigmoid

# def swish(x, beta = 1):
#     return (x * sigmoid(beta * x))

# # Getting the Custom object and updating them
# from keras.utils.generic_utils import get_custom_objects
# from keras.layers import Activation
  
# # Below in place of swish you can take any custom key for the name 
# get_custom_objects().update({'swish': Activation(swish)})




# model = load_model(f'./Models/{type}_models/q_{Q}/{type}Generation_forecast_MainModel_Q_{Q}.h5', custom_objects = {'<lambda>': lambda y,f: defined_loss(q,y,f), 'attention': attention, 'Activation': Activation(swish)})



# # split test data into sequences
# f = h5py.File(f"./Data/{type}/Processed_Data/{dataset_name}", "r")


# set_type = 'train'

# X_train1 = f[f'{set_type}_set'][f'X1_{set_type}'][1000:1845]
# X_train2 = f[f'{set_type}_set'][f'X2_{set_type}'][1000:1845]
# X_train3 = f[f'{set_type}_set'][f'X3_{set_type}'][1000:1845]
# X_train4 = f[f'{set_type}_set'][f'X1_{set_type}'][1000:1845]
# y_train = f[f'{set_type}_set'][f'y_{set_type}'][1000:1845]



# input_start, output_start = 0, input_seq_size

# seqX1, seqX2, seqX3, seqX4, seqY = [], [], [], [], []

# # a = np.array(X_train1)

# while (output_start + output_seq_size) <= len(y_train):
# 	# offset handled during pre-processing
# 	input_end = input_start + input_seq_size
# 	output_end = output_start + output_seq_size

# 	# inputs
# 	seqX1.append(X_train1[input_start:input_end])
# 	seqX2.append(X_train2[input_start:input_end])

# 	# outputs
# 	seqX3.append(X_train3[output_start:output_end])
# 	a = X_train4[output_start:output_end][:,:,:,1:]
# 	a = np.average(a, axis=(1,2))
# 	seqX4.append(a)
# 	seqY.append(y_train[output_start:output_end])

# 	input_start += output_seq_size
# 	output_start += output_seq_size


# x1, x2, x3, x4, y = np.array(seqX1), np.array(seqX2), np.array(seqX3), np.array(seqX4), np.array(seqY)
# f.close() 


# total_pred = []

# print(x1.shape)
# print(x2.shape)
# print(x3.shape)
# print(x4.shape)
# print(y.shape)


# for idx in range(len(x1)):
# 	print(idx)
# 	predictions = model.predict([x1[idx:idx+1, :, :, :, :], x2[idx:idx+1, :, :], x3[idx:idx+1, :, :], x4[idx:idx+1, :, :]])
# 	total_pred.append(predictions[0])

# total_pred = np.squeeze(np.concatenate(total_pred, axis=0))

# print(total_pred.shape)

# plt.plot(total_pred, label="prediction9")
# plt.plot(y.flatten(), label="actual")
# plt.show()


# exit()

################################################################################################################

#load training data dictionary
train_set_load = open("./Models/wind_models/predictions_0.1.pkl", "rb") 
train_set_01 = load(train_set_load)
train_set_load.close()


#load training data dictionary
# train_set_load = open("./Models/wind_models/predictions_0.5.pkl", "rb") 
# train_set_05 = load(train_set_load)
# train_set_load.close()



#load training data dictionary
train_set_load = open("./Models/wind_models/predictions_0.9.pkl", "rb") 
train_set_09 = load(train_set_load)
train_set_load.close()



f = h5py.File(f"./Data/wind/Processed_Data/train_set_V100.hdf5", "r")
y = f['train_set']['y_train']

print(y.shape)
print(train_set_09.shape)


idx = 820
plt.plot(train_set_01[idx:idx+168,0,0], label="prediction1")
# plt.plot(train_set_05[idx:idx+168,0,0], label="prediction5")
plt.plot(train_set_09[idx:idx+168,0,0], label="prediction9")
plt.plot(y[idx+672:idx+672+168,0].flatten(), label="actual")
plt.legend()
plt.show()


