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

def BMRS_Demand(**kwargs):

	url = 'https://api.bmreports.com/BMRS/{report}/V1?APIKey={API_KEY}&ServiceType=xml'.format(**kwargs)

	for key, value in kwargs.items():
		if key not in ['report']:
			a = "&%s=%s" % (key, value)
			url = url + a
	xml = objectify.parse(urllib.request.urlopen(url))

	tags = []
	output = []

	for root in xml.findall("./responseBody/responseList/item/"):
		tags.append(root.tag)
	tag = list(OrderedDict((x, 1) for x in tags).keys())
	df = pd.DataFrame(columns=tag)

	for root in xml.findall("./responseBody/responseList/item"):
		data = root.getchildren()
		output.append(data)
	df = pd.DataFrame(output, columns=tag)

	df = df.sort_values('settlementPeriod')
	
	return df


start_date = '2015-01-01'
end_date = '2015-01-03'

complete_df = pd.DataFrame()
daterange = pd.date_range(start_date, end_date)


for single_date in daterange:
	print(single_date.strftime("%Y-%m-%d"))
	df = BMRS_Demand(report='B0610', SettlementDate=single_date.strftime("%Y-%m-%d"), Period='*')
	complete_df = complete_df.append(df)

complete_df = complete_df[['settlementDate','settlementPeriod', 'quantity']]

# covnvert SPs to ints for concat
complete_df['settlementPeriod'] = complete_df['settlementPeriod'].astype('int64')


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
complete_df = complete_df.merge(time_df, on='settlementPeriod',how='left') 

complete_df['datetime'] = complete_df['settlementDate'] + " " + complete_df['settlementTimes']


complete_df = complete_df[['datetime', 'settlementPeriod', 'quantity']]

complete_df.to_clipboard()

pwd = os.getcwd()
complete_df.to_csv(pwd + '/' + "UK_Demand" + '_' + start_date + '_' + end_date + '.csv')













