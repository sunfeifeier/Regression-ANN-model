# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:50:31 2019

@author: lidon
"""


from pandas import Series
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
from pandas import read_csv

# load data
series = read_csv('Boston_Wind_Dataset.csv',index_col=0)
series = series["Label_year_data"]
series = series[-240:]
# prepare data
X = series.values
train, test = X[0:-24], X[-24:]
persistence_values = range(1, 25)
scores = list()
for p in persistence_values:
	# walk-forward validation
	history = [x for x in train]
	predictions = list()
	for i in range(len(test)):
		# make prediction
		yhat = history[-p]
		predictions.append(yhat)
		# observation
		history.append(test[i])
	# report performance
	rmse = sqrt(mean_squared_error(test, predictions))
	scores.append(rmse)
	print('p=%d RMSE:%.3f' % (p, rmse))
# plot scores over persistence values
pyplot.plot(persistence_values, scores)
pyplot.show()
 



####################
# Create lagged dataset
series = series[-120:]
values = DataFrame(series.values)

dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t', 't+1']
print(dataframe.head(5))
 
# split into train and test sets
X = dataframe.values
train_size = len(series)-72

train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
 
# persistence model
def model_persistence(x):
	return x
 
# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
 
# plot predictions and expected results
pyplot.plot(train_y,linewidth=1,c="r")
pyplot.plot([None for i in train_y] + [x for x in test_y],linewidth=1,c="r",label='Measured')
pyplot.plot([None for i in train_y] + [x for x in predictions],linewidth=1.5,
            label='Predicted',linestyle=':',c="b")
pyplot.legend(loc='upper left')
pyplot.show()
