import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pickle import dump
import h5py

from workalendar.europe import UnitedKingdom
cal = UnitedKingdom()

# load input data
windGen_data = pd.read_csv('../../data/raw/wind_labels/HH_windGen_uncleaned.csv', parse_dates=True, index_col=0, header=0, dayfirst=True)
solarGen_data = pd.read_csv('../../data/raw/solar_labels/HH_PVGen.csv', parse_dates=True, index_col=0, header=0, dayfirst=True)
demand_data = pd.read_csv('../../data/raw/demand_labels/HH_demand.csv', parse_dates=True, index_col=0, header=0, dayfirst=True)

# load labels
price_data = pd.read_csv('../../data/raw/price_labels/N2EX_UK_DA_Auction_Hourly_Prices.csv', parse_dates=True, index_col=0, header=0, dayfirst=True)

# interpolate hourly prices into HH resolution
price_data = price_data.reindex(pd.date_range(start=price_data.index.min(), end=price_data.index.max() + pd.Timedelta(minutes=30), freq='30T'))  
price_data = price_data.interpolate()

# combine vars into feature array
arrays = [windGen_data.values, solarGen_data.values, demand_data.values]

feature_array = []

# normalise feature array
for i, array in enumerate(arrays):
	scaler = MinMaxScaler()
	feature_array.append(scaler.fit_transform(array))

# normalise labels
scaler = MinMaxScaler() #normalise data
price_data = scaler.fit_transform(price_data.values)

# save price data scaler
dump(scaler, open('../../data/processed/price/_scaler/scaler_price.pkl', 'wb'))

# stack features
feature_array = np.concatenate(feature_array, axis=-1)

# mask data (eliminate nans)
wind_mask = windGen_data.iloc[:,-1].isna().groupby(windGen_data.index.normalize()).transform('any')
solar_mask = solarGen_data.iloc[:,-1].isna().groupby(solarGen_data.index.normalize()).transform('any')
demand_mask = demand_data.iloc[:,-1].isna().groupby(demand_data.index.normalize()).transform('any')
price_mask = demand_data.iloc[:,-1].isna().groupby(demand_data.index.normalize()).transform('any')

# eliminate all missing values with common mask
mask_all = wind_mask | solar_mask | demand_mask | price_mask

# apply mask, removing days with more than one nan value
feature_array = feature_array[~mask_all]

price_data = price_data[~mask_all]

# combine price data to other features for complete feature array
feature_array = [feature_array, price_data]
feature_array = np.concatenate(feature_array, axis=-1)

# time refs
time_refs = windGen_data.index
time_refs = time_refs[~mask_all]

# time data engineering 
df_times_outputs = pd.DataFrame()
df_times_outputs['date'] = time_refs.date
df_times_outputs['hour'] = time_refs.hour
df_times_outputs['month'] = time_refs.month - 1
df_times_outputs['year'] = time_refs.year
df_times_outputs['day_of_week'] = time_refs.dayofweek
df_times_outputs['day_of_year'] = time_refs.dayofyear - 1
df_times_outputs['weekend'] = df_times_outputs['day_of_week'].apply(lambda x: 1 if x>=5 else 0)

# account for bank / public holidays
start_date = time_refs.min()
end_date = time_refs.max()
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

# create sin / cos of output hour
times_out_hour_sin = np.expand_dims(np.sin(2*np.pi*df_times_outputs['hour']/np.max(df_times_outputs['hour'])), axis=-1)
times_out_hour_cos = np.expand_dims(np.cos(2*np.pi*df_times_outputs['hour']/np.max(df_times_outputs['hour'])), axis=-1)

# create sin / cos of output month
times_out_month_sin = np.expand_dims(np.sin(2*np.pi*df_times_outputs['month']/np.max(df_times_outputs['month'])), axis=-1)
times_out_month_cos = np.expand_dims(np.cos(2*np.pi*df_times_outputs['month']/np.max(df_times_outputs['month'])), axis=-1)

# create sin / cos of output year
times_out_year = np.expand_dims((df_times_outputs['year'].values - np.min(df_times_outputs['year'])) / (np.max(df_times_outputs['year']) - np.min(df_times_outputs['year'])), axis=-1)

# create sin / cos of output day of week
times_out_DoW_sin = np.expand_dims(np.sin(2*np.pi*df_times_outputs['day_of_week']/np.max(df_times_outputs['day_of_week'])), axis=-1)
times_out_DoW_cos = np.expand_dims(np.cos(2*np.pi*df_times_outputs['day_of_week']/np.max(df_times_outputs['day_of_week'])), axis=-1)

# create sin / cos of output day of year
times_out_DoY_sin = np.expand_dims(np.sin(2*np.pi*df_times_outputs['day_of_year']/np.max(df_times_outputs['day_of_year'])), axis=-1)
times_out_DoY_cos = np.expand_dims(np.cos(2*np.pi*df_times_outputs['day_of_year']/np.max(df_times_outputs['day_of_year'])), axis=-1)		

weekends = np.expand_dims(df_times_outputs['weekend'].values, axis =-1)
holidays = np.expand_dims(df_times_outputs['holiday'].values, axis =-1)

time_features = np.concatenate((times_out_hour_sin, times_out_hour_cos, times_out_month_sin, times_out_month_cos, times_out_DoW_sin, times_out_DoW_cos,
								 times_out_DoY_sin, times_out_DoY_cos, times_out_year, weekends, holidays), axis=-1)

# combine demand / solar / wind with time features
# combined_data = np.concatenate([feature_array, output_times], axis=-1)

test_split_seq = 9600 # larger test test to compensate adverse demand from COVID

# split data into train and test sets
dataset = {
	'train_set' : {
		'X1_train': feature_array[:-test_split_seq],
		'X2_train': time_features[:-test_split_seq],
		'X3_train': time_features[:-test_split_seq],
		'y_train': price_data[:-test_split_seq] 
		},
	'test_set' : {
		'X1_test': feature_array[-test_split_seq:],
		'X2_test': time_features[-test_split_seq:],
		'X3_test': time_features[-test_split_seq:],
		'y_test': price_data[-test_split_seq:] 
		}
	}

time_refs = {
	'input_times_train': time_refs[:-test_split_seq],
	'input_times_test': time_refs[-test_split_seq:], 
	'output_times_train': time_refs[:-test_split_seq],
	'output_times_test': time_refs[-test_split_seq:]
}

# print data for info
print(*[f'{key}: {dataset["train_set"][key].shape}' for key in dataset['train_set'].keys()], sep='\n')
print(*[f'{key}: {dataset["test_set"][key].shape}' for key in dataset['test_set'].keys()], sep='\n')

# save dataset
with open("../../data/processed/price/time_refs_price.pkl", "wb") as times:
	dump(time_refs, times)

# save training set as dictionary (h5py dump)
f = h5py.File('../../data/processed/price/dataset_price.hdf5', 'w')

for group_name in dataset:
	group = f.create_group(group_name)
	for dset_name in dataset[group_name]:
		dset = group.create_dataset(dset_name, data = dataset[group_name][dset_name])
f.close()










