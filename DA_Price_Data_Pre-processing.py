import pandas as pd
import numpy as np






# load input data
windGen_data = pd.read_csv('./Data/wind/Raw_Data/HH_windGen_V4.csv', parse_dates=True, index_col=0, header=0, dayfirst=True)
solarGen_data = pd.read_csv('./Data/solar/Raw_Data/HH_PVGeneration_v2.csv', parse_dates=True, index_col=0, header=0, dayfirst=True)
demand_data = pd.read_csv('./Data/demand/Raw_Data/uk_demand_20190101_20210630.csv', parse_dates=True, index_col=0, header=0, dayfirst=True)


# combine vars into feature array
feature_array = [windGen_data.values, solarGen_data.values, demand_data.values]

print(windGen_data.values.shape)
print(solarGen_data.values.shape)
print(demand_data.values.shape)

# stack features
feature_array = np.stack(feature_array, axis = -1)

# normalise feature array
for i, array in enumerate(feature_array):
	scaler = StandardScaler(with_mean=False) #normalise data
	feature_array[i] = scaler.fit_transform(array)


# mask data (eliminate nans)
wind_mask = windGen_data['MW'].isna().groupby(windGen_data.index.normalize()).transform('any')
solar_mask = solarGen_data['quantity (MW)'].isna().groupby(solarGen_data.index.normalize()).transform('any')
demand_mask = demand_data['quantity (MW)'].isna().groupby(demand_data.index.normalize()).transform('any')

# eliminate all missing values with common mask
mask_all = wind_mask + solar_mask + demand_mask

# apply mask, removing days with more than one nan value
feature_array = feature_array[~outputs_mask]
labels = labels[~outputs_mask]



# time data engineering 
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


output_times = np.concatenate((times_out_hour_sin, times_out_hour_cos, times_out_month_sin, times_out_month_cos, times_out_DoW_sin, times_out_DoW_cos,
								 times_out_DoY_sin, times_out_DoY_cos, times_out_year, weekends, holidays), axis=-1)




# save dataset
