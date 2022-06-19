# format prediction results for qunatile forecasting d3 plot
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import csv
from pickle import load
from sklearn.preprocessing import MinMaxScaler



# declare model type
model_type = 'seq2seq+temporal' # - bilstm, seq2seq, seq2seq+temporal, seq2seq+temporal+spatial

# forecasting model
forecast_var = 'price' 

# select start example index reference, 7-days plotted from here
ex_idx = 4

# load prediction data
with open(f'../../results/{forecast_var}/{model_type}/forecasted_time_series_{forecast_var}_{model_type}.pkl', 'rb') as forecast_data:
	predictions = load(forecast_data)

print(len(predictions['0.5']))	

# get start date  
out_start_time = predictions['time_refs']['output_times'][ex_idx][0]

print(out_start_time)

# produce date range for week-long predictions
ouput_sequence_len = 336 # (Half-Hours)
input_num_of_days = ouput_sequence_len / 48
# start_date = datetime.strptime(str(out_start_time)[:10], "%Y-%m-%d")
# out_date_range = pd.date_range(start=start_date, end=start_date + timedelta(days=input_num_of_days) , freq="30min")[:-1]# remove HH entry form unwanted day

out_start_time = predictions['time_refs']['output_times'][ex_idx:ex_idx+int(input_num_of_days)]
out_date_range = pd.to_datetime(out_start_time.ravel(), format='%Y-%m-%d')

# index ref
idx_ref = [x for x in range(1, ouput_sequence_len+1)]

# final params for df 
final_params = {'year': idx_ref, 
				'Datetime': out_date_range }

# loop through to write results for each quantile
for q in list(predictions.keys())[:-2]:

	final_params[f'q_{q[2:]}'] = predictions[str(q)][ex_idx:ex_idx+7, :, 0].reshape((-1))

# add actual values for reference
final_params['actual'] = predictions['y_true'][ex_idx:ex_idx+7, :, 0].reshape((-1))

print(final_params.keys())

# convert to pandas df
df = pd.DataFrame(dict([(keys ,pd.Series(values, dtype = 'object')) for keys, values in final_params.items()])) # set all as objects to avoid warning on empty cells

# divide to GW
if forecast_var != "price":
	df.iloc[:,2:] = df.iloc[:,2:] / 1000

# copy to clipboard
df.to_clipboard()

# save data to file
df.to_csv(f'../../results/{forecast_var}/{model_type}/quantile_prediction_results_{forecast_var}_{model_type}.csv', index=False)

