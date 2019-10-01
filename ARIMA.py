#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:07:53 2017
###https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/
https://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/
articel"Neural network forecasting for seasonal and trend time series"
https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
https://machinelearningmastery.com/make-sample-forecasts-arima-python/
https://machinelearningmastery.com/time-series-data-visualization-with-python/
https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

https://machinelearningmastery.com/simple-time-series-forecasting-models/
@author: fei
"""
#
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import warnings
from math import sqrt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 5, 8
from pandas import Series,DataFrame
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller,acf, pacf
from pandas.core import datetools
import pandas.tseries
#import pandas.core.DatetimeIndex
from numpy import log
from datetime import datetime,timedelta
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.tools.plotting import autocorrelation_plot
from pandas.plotting import autocorrelation_plot

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pandas.tools.plotting import lag_plot


#import scikits.statsmodels.tsa.arima_process as ap

df=read_csv("Boston_Wind_Dataset.csv")
#series=Series.from_csv("Boston_Wind_Dataset.csv")
series = df["Label_year_data"]
series=series[-120:]
x=series.values

def difference(dataset, interval):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

stationary=difference(x,1)
#diff_series= series - series.shift(12) #seame function
# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

"""
p-value > 0.05: Accept the null hypothesis (H0), 
the data has a unit root and is non-stationary.
p-value <= 0.05: Reject the null hypothesis (H0), 
the data does not have a unit root and is stationary.
The more negative this statistic, the more likely we are to reject the null
hypothesis (we have a stationary dataset). 
rejecting the null hypothesis means that the process has no unit root,
and in turn that the time series is stationary or does not have 
time-dependent structure


#two times difference period[12,1] shift and diff are the same function
diff_12 = series.diff(12)
diff_12.dropna(inplace=True)
diff_12_1 = diff_12.diff(1)
"""

autocorrelation_plot(x)
plt.title('Autocorrelation')
lag_plot(series)
pyplot.show()

# analysis
#ACF and PACF plots:

lag_acf = acf(stationary, nlags=60)
lag_pacf = pacf(stationary, nlags=60, method='ols')
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-0.2,linestyle='--',color='gray')
plt.axhline(y=0.2,linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#plt.axhline(y=-1.96/np.sqrt(len(diff)),linestyle='--',color='gray')
#plt.axhline(y=1.96/np.sqrt(len(diff)),linestyle='--',color='gray')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-0.2,linestyle='--',color='gray')
plt.axhline(y=0.2,linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


# multi-step out of sample forecasting with forecast function
#https://machinelearningmastery.com/make-sample-forecasts-arima-python/


model = ARIMA(differenced, order=(5,1,1))
model_fit = model.fit(disp=0)
# multi-step out-of-sample forecast
forecast = model_fit.forecast(steps=step)[0]
# invert the differenced forecast to something usable
history = [x for x in X]
#cost_history=np.empty(shape=[1],dtype=float)
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, Hours_in_day)
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1
inv_yhat=history[-step:]
np.array(inv_yhat)
np.array(validation)
for i in range(len(validation)):
    i+=1
    error = mean_squared_error(inv_yhat, validation)
    rmse = sqrt(mean_squared_error(inv_yhat, validation))
    print('Test MSE: %.3f' % error,'Test RMSE: %.3f' % rmse)
  



# evaluate combinations of p, d and q values for an ARIMA model
#https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/
#https://stackoverflow.com/questions/22770352/auto-arima-equivalent-for-python
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error
 
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse=evaluate_arima_model(dataset, order)
                      aic=evaluate_arima_model.aic
                      bic=evaluate_arima_model.bic
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
# evaluate parameters
p_values = [0,4,5,6,8,10]
d_values = range(0, 3)
q_values = range(0, 5)
warnings.filterwarnings("ignore")
series=Series.from_csv("10year_winddata_practice.csv",header=0)
series=series[-760:]
evaluate_models(series.values, p_values, d_values, q_values)



# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())

#In this plot, the two dotted lines on either sides of 0 are the confidence interevals. 
#these can be used to determine the ‘p’ and ‘q’ values as:
#1.p – The lag value where the PACF chart crosses the upper confidence interval 
#for the first time. If you notice closely, in this case p=2.
#2.q – The lag value where the ACF chart crosses the upper confidence interval 
#for the first time. If you notice closely, in this case q=2.


#Convert to original scale:
#determine the cumulative sum at index and then add it to the base number.
#predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
#predictions_ARIMA_diff_cumsum.head()

#https://machinelearningmastery.com/time-series-seasonality-with-python/
#https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
#https://machinelearningmastery.com/seasonal-persistence-forecasting-python/


ARIMA(0, 0, 0) MSE=10.859
ARIMA(0, 0, 1) MSE=4.521
ARIMA(0, 0, 2) MSE=2.742
ARIMA(0, 0, 3) MSE=2.261
ARIMA(0, 0, 4) MSE=1.869
ARIMA(0, 1, 0) MSE=1.458
ARIMA(0, 1, 1) MSE=1.451
ARIMA(0, 1, 2) MSE=1.461
ARIMA(0, 1, 3) MSE=1.463
ARIMA(0, 1, 4) MSE=1.464
ARIMA(0, 2, 0) MSE=3.191
ARIMA(0, 2, 1) MSE=1.465
ARIMA(4, 0, 0) MSE=1.396
ARIMA(4, 0, 1) MSE=1.403
ARIMA(4, 0, 3) MSE=1.401
ARIMA(4, 0, 4) MSE=1.384
ARIMA(4, 1, 0) MSE=1.463
ARIMA(4, 1, 1) MSE=1.462
ARIMA(4, 1, 3) MSE=1.414
ARIMA(4, 2, 0) MSE=1.743
ARIMA(5, 0, 0) MSE=1.383
ARIMA(5, 0, 1) MSE=1.397
ARIMA(5, 0, 2) MSE=1.398
ARIMA(5, 0, 4) MSE=1.396
ARIMA(5, 1, 0) MSE=1.461
ARIMA(5, 1, 2) MSE=1.453
ARIMA(5, 2, 0) MSE=1.720
ARIMA(6, 0, 0) MSE=1.389
ARIMA(6, 0, 1) MSE=1.390
ARIMA(6, 0, 3) MSE=1.389
ARIMA(6, 1, 0) MSE=1.459
ARIMA(6, 2, 0) MSE=1.696
ARIMA(8, 0, 0) MSE=1.396
ARIMA(8, 0, 1) MSE=1.393
ARIMA(8, 0, 2) MSE=1.412
ARIMA(8, 1, 0) MSE=1.446
ARIMA(8, 1, 1) MSE=1.448
ARIMA(8, 2, 0) MSE=1.672
ARIMA(8, 2, 1) MSE=1.453
ARIMA(10, 0, 0) MSE=1.402
ARIMA(10, 0, 1) MSE=1.403
ARIMA(10, 0, 2) MSE=1.405
ARIMA(10, 0, 3) MSE=1.408
ARIMA(10, 0, 4) MSE=1.395
ARIMA(10, 1, 0) MSE=1.452
ARIMA(10, 1, 1) MSE=1.384
ARIMA(10, 2, 0) MSE=1.661
ARIMA(10, 2, 1) MSE=1.459
Best ARIMA(5, 0, 0) MSE=1.383




# Recursive forecast ontstep and rolling forecast
#https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/


x = series.values
Hours_in_day=1
#differenced = difference(X, Hours_in_day)
size = 24
#train_1, test_1 = X[0:size], X[size:len(X)]
train, test = x[0:size], x[size:len(X)]
history = [x for x in train]
predictions = list()
obs=[x for x in test]
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
  yhat.tolist
	predictions.append(yhat)    
	#obs = test[t]
  #obs=yhat
	history.append(yhat)    
	print('predicted=%f, expected=%f' % (yhat, test[t]))

pred=history[size:]
for i in range(len(test)):
    i=+1
    error = mean_squared_error(pred, test)
    print('Test MSE: %.3f' % error)
   
print(model_fit.summary())
# plot
rcParams['figure.figsize'] = 15, 5
pyplot.plot(test,color="black")
pyplot.plot(predictions, color='red')
pyplot.show()
history.append(obs)
