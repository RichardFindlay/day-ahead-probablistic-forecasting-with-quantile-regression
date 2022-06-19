# format attention results for context d3 plot
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import csv
from pickle import load
from sklearn.preprocessing import MinMaxScaler

# forecasting model
type = 'solar' # 'wind', 'solar', 'price', 'demand'

# select example refernce
ex_idx = 26

# load attention data 
if type != "price":
	with open(f'../../results/{type}/seq2seq+temporal+spatial/attention_data_{type}_seq2seq+temporal+spatial.pkl', 'rb') as attention_data:
		attention_results = load(attention_data)

	# load prediction data
	with open(f'../../results/{type}/seq2seq+temporal+spatial/forecasted_time_series_{type}_seq2seq+temporal+spatial.pkl', 'rb') as forecast_data:
		predictions = load(forecast_data)
else:
	with open(f'../../results/{type}/seq2seq+temporal/attention_data_{type}_seq2seq+temporal.pkl', 'rb') as attention_data:
		attention_results = load(attention_data)

	# load prediction data
	with open(f'../../results/{type}/seq2seq+temporal/forecasted_time_series_{type}_seq2seq+temporal.pkl', 'rb') as forecast_data:
		predictions = load(forecast_data)


print(attention_results.keys())

# get start dates for inputs and outputs
in_start_time = attention_results['time_refs']['input_times'][ex_idx][0]
out_start_time = attention_results['time_refs']['output_times'][ex_idx][0]

# log start date of selected index
print(f'input time start date: {in_start_time}')
print(f'output time start date: {out_start_time}')

# input data for reference
if type != 'price':
	input_data = np.average(attention_results['input_features'][ex_idx, :, :, :, 0], axis=(1,2))
else:
	input_data = attention_results['input_features'][ex_idx, :, -1:]


# get prediction result for current index
current_prediction = predictions['0.5'][ex_idx, :, 0]

# attention values for current index
current_attention_vals = attention_results['0.5'][ex_idx]

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

# out_start_time = predictions['time_refs']['output_times'][ex_idx:ex_idx+int(input_num_of_days)]
# input_date_range = pd.to_datetime(out_start_time.ravel(), format='%Y-%m-%d')

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

# load and apply scaler
# load scaler 
scaler = load(open(f'../../data/processed/{type}/_scaler/scaler_{type}_v2.pkl', 'rb'))

input_data = np.squeeze(input_data)

# transform input data
input_data = scaler.inverse_transform(input_data)

attention_vals_int = attention_vals

# take log of attention values
# scaler = MinMaxScaler(feature_range = (0, 1))
# attention_vals_scaled = scaler.fit_transform(attention_vals.reshape(-1,1)).reshape(-1)

# attention_vals_scaled = np.sqrt(attention_vals_scaled)

attention_vals_scaled = attention_vals

# get true values for reference
y_true = predictions['y_true'][ex_idx][:,0]

# final params for df 
final_params = {'group_index': group_index, 
				'variable_index': variable_index, 
				'group': group, 
				'variable': variable, 
				'value_scaled': attention_vals_scaled, 
				'value': attention_vals_int, 
				'input_time_ref': input_time_ref, 
				'input_time': input_time, 
				'input_values': input_data, 
				'output_time_ref': output_time_ref, 
				'output_time': output_time, 
				'prediction': current_prediction,
				'y_true': y_true }

# convert to pandas df
df = pd.DataFrame(dict([(keys ,pd.Series(values, dtype = 'object')) for keys, values in final_params.items()])) # set all as objects to avoid warning on empty cells

# copy to clipboard
df.to_clipboard()

# save data to file
# df.to_csv(f'../../results/{type}/attention_plot_results_{type}.csv', index=False)


