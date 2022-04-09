# format attention results for context plot
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import csv
from pickle import load
from sklearn.preprocessing import MinMaxScaler

# forecasting model
type = 'solar' 

# select example refernce
ex_idx = 40 

# load attention data 
with open(f'attention_data_{type}.pkl', 'rb') as attention_data:
	attention_results = load(attention_data)

# load prediction data
with open(f'forecasted_time_series_{type}.pkl', 'rb') as forecast_data:
	predictions = load(forecast_data)

# get start dates for inputs and outputs
in_start_time = attention_results['time_refs']['input_times'][ex_idx][0]
out_start_time = attention_results['time_refs']['output_times'][ex_idx][0]

# log start date of selected index
print(f'input time start date: {in_start_time}')
print(f'output time start date: {out_start_time}')

# input data for reference
input_data = np.average(attention_results['input_features'][ex_idx, :, :, :, 0], axis=(1,2))

# get prediction result for current index
current_prediction = predictions['0.5'][ex_idx, :, 0]

# attention values for current index
current_attention_vals = attention_results['0.5'][ex_idx]
print(current_attention_vals.shape)

attention_vals = np.empty((current_attention_vals.shape[0] * current_attention_vals.shape[1]))

# make sure attention values are in correct format
iidx = 0
for idx in range(current_attention_vals.shape[0]):
	attention_vals[iidx:iidx+48] = current_attention_vals[idx, :]
	iidx += 48



# input params
input_sequence_len = 336 
input_num_of_days = input_sequence_len / 48
start_date = datetime.strptime(str(in_start_time)[:10], "%Y-%m-%d")
target_data =  datetime.strptime(str(out_start_time)[:10], "%Y-%m-%d")
input_date_range = pd.date_range(start=start_date, end=start_date + timedelta(days=input_num_of_days) , freq="30min")[:-1]# remove HH entry form unwanted day

# create index values
group_index = [48 * [idx] for idx in range(input_sequence_len)]
variable_index = [[idx for idx in range(48)] for iidx in range(input_sequence_len)]

# flatten lists if lists
group_index = sum(group_index, [])
variable_index = sum(variable_index, [])

# create data ranges
group = [48 * [date_time] for date_time in input_date_range]
variable =  [pd.date_range(start=target_data, end=target_data + timedelta(days=1) , freq="30min").tolist()[:-1] for idx in range(input_sequence_len)] # remove HH entry for next day

# flatten timestamps into single list
group = sum(group, [])
variable = sum(variable, [])

# create output time idxs
output_time_ref = [idx for idx in range(48)]

# create input time idxs
input_time_ref = [idx for idx in range(input_sequence_len)]

# input times
input_time = [date_time for date_time in input_date_range]

# output times
output_time =  pd.date_range(start=target_data, end=target_data + timedelta(days=1) , freq="30min").tolist()[:-1]


attention_vals_int = attention_vals

# take log of attention values
scaler = MinMaxScaler(feature_range = (0, 1))
attention_vals_scaled = scaler.fit_transform(attention_vals.reshape(-1,1)).reshape(-1)

attention_vals_scaled = np.sqrt(attention_vals_scaled)

# final params for df 
final_params = {'group_index': group_index, 
				'variable_index': variable_index, 
				'group': group, 
				'variable': variable, 
				'value_scaled': attention_vals_scaled, 
				'value': attention_vals_int, 
				'input_time_ref': input_time_ref, 
				'input_time': input_time, 
				'input_solar': input_data, 
				'input_cloud_cover': [],
				'output_time_ref': output_time_ref, 
				'output_time': output_time, 
				'prediction': current_prediction}

# convert to pandas df
df = pd.DataFrame(dict([(keys ,pd.Series(values, dtype = 'object')) for keys, values in final_params.items()])) # set all as objects to avoid warning on empty cells

df.to_clipboard()

df.to_csv(f'attention_plot_results_{type}.csv', index=False)


