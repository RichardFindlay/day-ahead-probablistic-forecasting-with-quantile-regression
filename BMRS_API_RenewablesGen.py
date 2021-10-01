#scrape Elexon API for data
#As copied from Patrick Avis (Energyanalyst.co.uk)

#import libraries
import numpy as np
import urllib
import urllib.request
import pandas as pd
from lxml import objectify
from collections import OrderedDict
import os


API_KEY = ...


def BMRS_GetXML(**kwargs):
	'''BMRS_XMLGet(api=**YOUR-API-KEY-HERE**, report='PHYBMDATA', sd='2016-01-26', sp=3,
	bmu='T_COTPS-1',bmutype='T', leadpartyname = 'AES New Energy Limited',ngcbmuname='EAS-ASP01')'''

	url = 'https://api.bmreports.com/BMRS/{report}/V1?APIKey={API_KEY}&ServiceType=xml'.format(**kwargs)

	for key, value in kwargs.items():
		if key not in ['report']:
			a = "&%s=%s" % (key, value)
			url = url + a
	# print(url)
	xml = objectify.parse(urllib.request.urlopen(url))

	print(url)
	
	return xml


def BMRS_Dataframe(**kwargs):
	'''Takes the sourced XML file produces a dataframe from all the children of the ITEM tag'''
	tags = []
	output = []

	for root in BMRS_GetXML(**kwargs).findall("./responseBody/responseList/item/"):
		tags.append(root.tag)
	tag = list(OrderedDict((x, 1) for x in tags).keys())
	df = pd.DataFrame(columns=tag)

	for root in BMRS_GetXML(**kwargs).findall("./responseBody/responseList/item"):
		data = root.getchildren()
		output.append(data)
	df = pd.DataFrame(output, columns=tag)
	


	
	df = df[['powerSystemResourceType', 'settlementDate', 'settlementPeriod', 'quantity']].copy()


	# divide df into onshore and offshore wind gen
	onshore_wind = df.loc[df['powerSystemResourceType'] == '"Wind Onshore"']
	offshore_wind = df.loc[df['powerSystemResourceType'] == '"Wind Offshore"']

	# convert to floats
	onshore_wind['quantity'] = onshore_wind['quantity'].astype('float32')
	offshore_wind['quantity'] = offshore_wind['quantity'].astype('float32')

	# make new df for total wind generation
	total_wind = onshore_wind.copy()
	del total_wind['powerSystemResourceType'] 

	# add onshore and offshore wind generation
	total_wind['quantity'] = onshore_wind['quantity'].values + offshore_wind['quantity'].values 

	# covnvert SPs to ints for concat
	total_wind['settlementPeriod'] = total_wind['settlementPeriod'].astype('int64')

	# create template time df to concatenate
	settlementTimes = []
	times = ["00", "30"]

	# create array of HHs
	for i in range(24):
		for j in range(len(times)):
			Times = str(i).zfill(2) + ":" + times[j] + ":00" 
			settlementTimes.append(Times)

	settlementInts = np.array([t+1 for t in range(48)])

	time_df = pd.DataFrame({'settlementPeriod':settlementInts, 'settlementTimes':settlementTimes})


	# concat times to dataframe for reference
	total_wind = total_wind.merge(time_df,on='settlementPeriod',how='left') 

	# sort in settlement periods in ascending order
	total_wind.sort_values('settlementPeriod', inplace = True)
	total_wind.reset_index(inplace=True)


	# combine date & time into one column
	total_wind['datetime'] = total_wind['settlementDate'] + " " + total_wind['settlementTimes'] 

	# remove unwanted columns
	final_df = total_wind[['datetime', 'quantity']].copy()


	return final_df


start_date = '2015-01-01'
end_date = '2021-06-30'

complete_df = pd.DataFrame()

daterange = pd.date_range(start_date, end_date)




for single_date in daterange:
	print(single_date.strftime("%Y-%m-%d"))
	df = BMRS_Dataframe(report='B1630', SettlementDate=single_date.strftime("%Y-%m-%d"), Period='*')
	complete_df = complete_df.append(df)


complete_df.to_clipboard()

pwd = os.getcwd()
complete_df.to_csv(pwd + '/' + "UK_Wind_Gen" + '_' + start_date + '_' + end_date + '.csv')












