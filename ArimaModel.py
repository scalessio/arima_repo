#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time


# In[2]:


from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from matplotlib import pyplot
import numpy
from numpy import array
import pandas as pd
from natsort import natsorted, ns
import csv
import glob, os
from statsmodels.tsa.arima_model import ARIMA


# In[3]:


def retrive():
    a=[]
    i=0
    for file in os.listdir('data/data/'):
        if file.endswith(".csv"):
            filename = file
            a.append(filename)
            a = [w.replace('.csv', '') for w in a]
    return(a,len(a))
#Define a function. Use the parser function for take the data as datetime, is a function of pandas library.
def parser(x,y,z):
    x =x+':'+y+':'+z
    return datetime.strptime(x,' %d/%b/%Y:%H:%M')
# load dataset
def load_data(vettore,leng):
    contr = False
    for x in range(leng):      
        day = vettore[x]
        #print(vettore[x])
        if contr == True:
            series_temp = read_csv('data/data/%s.csv'%day, header=0,
                          parse_dates={'date_time' :['Day','Hour','Minute']}, index_col = 'date_time',
                          squeeze=True, date_parser=parser)
            series_temp = series_temp[series_temp['Byte_count'] != 0]
            series_temp = series_temp[:-1]
            series_temp.head(2)
            series = series.append(series_temp)
            
            counter = counter + len(series_temp)
            
        else:
            series = read_csv('data/data/%s.csv'%day, header=0,
                          parse_dates={'date_time' :['Day','Hour','Minute']}, index_col = 'date_time',
                          squeeze=True, date_parser=parser)
            counter = len(series)
            
            contr = True
    return series


<<<<<<< HEAD
# In[20]:



def runs(exp,lag):
        filename = 'arima_forecast_N%d_Lg%d.csv'%(exp,lag)
        vet,leng = retrive()
        vet = natsorted(vet, key=lambda y: y.lower())
        vet
        series = load_data(vet,leng)
        test_samples = 600
        series =  series[['Byte_count','Request_count']]
        series = series.Request_count
        ser_ind = series
        series = series.values
        train = series[:-test_samples]
        test = series[-test_samples:]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
	        model = ARIMA(history, order=(lag,1,0))
	        model_fit = model.fit(disp=0)
	        output = model_fit.forecast()
	        yhat = output[0]
	        predictions.append(yhat)
	        obs = test[t]
	        history.append(obs)
	        print('predicted=%f, expected=%f' % (yhat, obs))
        s = []
        for x in range(0,len(predictions)):
            s = numpy.append(s,predictions[x])
        error = sqrt(mean_squared_error(test, s))
        print('Test RMSE: %.3f' % error)
        pyplot.rcParams['figure.figsize'] = (12,9)
        pyplot.plot(test)
        pyplot.plot(predictions, color='red')
        pyplot.xlabel('Minutes',fontsize = 14)
        pyplot.ylabel('Request Count',fontsize = 14)
        pyplot.title('Arima Vs Ground Truth Forecast', weight='bold',fontsize = 20)
        pyplot.legend(loc='upper left', fancybox=True, fontsize='large', framealpha=0.5) 
        pyplot.savefig('plots/ArimaVSthrut.png')
        pyplot.show()
        idx = ser_ind.tail(6000)
        idx = idx.index
        truth_prediction = pd.DataFrame(index=idx)
        arima_prediction = pd.DataFrame( index=idx)


        truth_prediction['t']=test[:]
        arima_prediction['t']=s[:]


        arima_prediction.to_csv('results/Eval/'+filename, sep='\t', encoding='utf-8')
        truth_prediction.to_csv('test_prediction.csv', sep='\t', encoding='utf-8')

for x in range(1,6):
        
        time_start = time.time()
        lag=5
        exp=x
        print("Run Arima exp n%d"%exp)
        runs(exp,lag)
        time_end = time.time()
        duration = time_end-time_start
        rows = ['Start','End','Duration']
        timedf= DataFrame(columns=['Time'],index=rows)
        timedf.iloc[0]=time_start
        timedf.iloc[1]=time_end
        timedf.iloc[2]=duration
        timedf.to_csv('Duration_ARIMA_600_exp%d.csv'%exp, sep='\t',encoding='utf-8')
=======
# In[ ]:



time_start = time.time()
vet,leng = retrive()
vet = natsorted(vet, key=lambda y: y.lower())
vet
series = load_data(vet,leng)
test_samples = 600
series =  series[['Byte_count','Request_count']]
series = series.Request_count
ser_ind = series
series = series.values
train = series[:-test_samples]
test = series[-test_samples:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(60,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print("sample %d"%t)
	print('predicted=%f, expected=%f' % (yhat, obs))
s = []
for x in range(0,len(predictions)):
    s = numpy.append(s,predictions[x])
error = sqrt(mean_squared_error(test, s))
print('Test RMSE: %.3f' % error)
pyplot.rcParams['figure.figsize'] = (12,9)
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.xlabel('Minutes',fontsize = 14)
pyplot.ylabel('Request Count',fontsize = 14)
pyplot.title('Arima Vs Ground Truth Forecast', weight='bold',fontsize = 20)
pyplot.legend(loc='upper left', fancybox=True, fontsize='large', framealpha=0.5) 
pyplot.savefig('plots/ArimaVSthrut.png')
pyplot.show()
idx = ser_ind.tail(600)
idx = idx.index
truth_prediction = pd.DataFrame(index=idx)
arima_prediction = pd.DataFrame( index=idx)


truth_prediction['t']=test[:]
arima_prediction['t']=s[:]
     
arima_prediction.to_csv('arima_prediction600tst_lg5.csv', sep='\t', encoding='utf-8')
truth_prediction.to_csv('test_prediction600_lg5.csv', sep='\t', encoding='utf-8')
time_end = time.time()
duration = time_end-time_start
rows = ['Start','End','Duration']
timedf= DataFrame(columns=['Time'],index=rows)
timedf.iloc[0]=time_start
timedf.iloc[1]=time_end
timedf.iloc[2]=duration
timedf.to_csv('Duration60Lag.csv', sep='\t',encoding='utf-8')
>>>>>>> 6333b44bede242e707e8a772ab40d292d4619f87


# In[ ]:




