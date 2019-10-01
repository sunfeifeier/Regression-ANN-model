# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:08:38 2018

@author: lidon
"""

from pandas import read_csv
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler, Normalizer
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from math import sqrt
import pandas as pd
import numpy as np
from keras import optimizers 
from numpy import concatenate
import matplotlib.pyplot as plt
#from ann_visualizer.visualize import ann_viz
from keras import losses
#import seaborn as sns
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from pandas import concat
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras import regularizers
from keras.regularizers import l1

df=read_csv("Boston_Wind_Dataset.csv")
data1=df["Label_year_data"]
data1 = pd.DataFrame(data1)
data1=data1.replace(to_replace=0,value=1)

n_seq = 24
n_lag=5
time_step=1
n_batch = 1
n_test=240
n_val=760
n_hidden_1=4
n_hidden_2=3
n_hidden_3=4
n_hidden_4=4
#learning_rate=0.0005
training_epochs=10

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
data2=series_to_supervised(data1,n_lag,n_seq)
#print(data2.head())

#Feature scaling
scaler_1 = MinMaxScaler(feature_range=(0, 1))
scaler_2 = MinMaxScaler(feature_range=(0, 1))
#scaler_1 = StandardScaler()
#scaler_2 = StandardScaler()
#scaler = MinMaxScaler(feature_range=(0, 1))
scale_X =data2.iloc[:,0:n_lag]
scale_Y =data2.iloc[:,n_lag:(n_seq+n_lag+1)]
scalerX = scaler_1.fit(scale_X)
scalery = scaler_2.fit(scale_Y)

scaled_X = scalerX.transform(scale_X)
scaled_Y = scalery.transform(scale_Y)
scaled_Y = scaled_Y.astype('float32')
scaled_X = scaled_X.astype('float32')

X_train, X_test ,X_val= scaled_X[:(len(scaled_X)-n_val-n_test)], scaled_X[-n_test:], scaled_X[-(n_val+n_test):-n_test]
y_train, y_test,y_val = scaled_Y[:(len(scaled_X)-n_val-n_test)], scaled_Y[-n_test:],scaled_Y[-(n_val+n_test):-n_test]
#y_train, y_test,y_val = scaled_Y[:(len(scaled_X)-n_val-n_test)], scaled_Y[-n_test:],scaled_Y[n_val:(n_test+n_val)]


X_train=X_train.reshape((X_train.shape[0],time_step, X_train.shape[1]))
#X_train=np.reshape(X_train,(X_train.shape[0],time_step, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], time_step, X_test.shape[1]))
X_val = X_val.reshape((X_val.shape[0], time_step, X_val.shape[1]))


# two hidden layers model
model = Sequential()
model.add(LSTM(n_hidden_1,return_sequences=True,activation='linear',stateful=True,
               activity_regularizer=l1(0.001),
               batch_input_shape=(n_batch, time_step, X_train.shape[2])))
#dropout Fraction of the units to drop for the linear transformation of the inputs.
model.add(Activation('relu'))
model.add(LSTM(n_hidden_2,batch_input_shape=(n_batch, time_step, X_train.shape[2]), activation='relu',stateful=True))
model.add(Dense(y_train.shape[1]))
#model.add(TimeDistributed(Dense(y_train.shape[1])))
#sgd = optimizers.SGD(lr=1, nesterov=True)
adam=optimizers.Adam(lr=0.0005)

model.compile(loss=losses.mean_squared_error, optimizer="adam")
#model.compile(loss=losses.mean_absolute_percentage_error,optimizer="adam")
checkpoint = ModelCheckpoint(filepath="best_weights_mod5.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


#model.summary()
#model.input_shape
history = model.fit(X_train, y_train, validation_data=(X_val,y_val),callbacks=callbacks_list, epochs=10, batch_size=n_batch, verbose=2,shuffle=False)

# plot train and validation loss
plt.plot(history.history['loss'][30:])
plt.plot(history.history['val_loss'][30:])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

###################################################
verbose=1
losses = []
val_losses = []
min_val_loss = (99999,999999)
for i in range(training_epochs):
    if verbose!=0:
        print(i)
    history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=2, batch_size=n_batch, verbose=2, shuffle=False)
    losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'][0])
    if val_losses[-1] < min_val_loss[0]:
        min_val_loss = (val_losses[-1], i)
#    model.reset_states()
print('best val_loss and epoch:',min_val_loss)
plt.title('loss')
plt.plot(losses,color="blue")
plt.plot(val_losses, color='red')
plt.show()


# forecasting
pred_y = model.predict(X_test,batch_size=n_batch)
inv_yhat = scalery.inverse_transform(pred_y)
real_test=scale_Y[-n_test:]




# calculate MSE with scaled 
mse = 0
rmse = 0
for i in range(n_seq):
    rmse = sqrt(mean_squared_error(y_test, pred_y))   
    mse = mean_squared_error(y_test, pred_y)
    MAE=mean_absolute_error(y_test, pred_y)
#    MAPE=mean_absolute_percentage_error(test_y, pred_y)
    print('t+%d RMSE: %f' % ((i+1), rmse)) 
    print('t+%d MAE: %f' % ((i+1), MAE)) 
    print('t+%d MSE: %f' % ((i+1), mse))

#with inversed value
for i in range(n_seq):
    rmse = sqrt(mean_squared_error(real_test, inv_yhat))   
    mse = mean_squared_error(real_test, inv_yhat)
    MAE=mean_absolute_error(real_test, inv_yhat)
#    MAPE=MAE/real_test
    print('t+%d RMSE: %f' % ((i+1), rmse))   
    print('t+%d MSE: %f' % ((i+1), mse))
    print('t+%d MAE: %f' % ((i+1), MAE))
#    print('t+%d MAPE: %f' % ((i+1), MAPE))


# evaluate the MSE for each forecast time step
mse = 0
rmse = 0
for i in range(len(y_test)):
    actual = y_test[i]
    predicted = forecast[i]
    a_rmse = sqrt(mean_squared_error(actual, predicted))   
    b_mse = mean_squared_error(actual, predicted)
    mse+=b_mse
    rmse+=a_rmse  
print('RMSE: %f' %  (rmse/8760))   
print('MSE: %f' % (mse/8760))


#### 二层 hidden Layers model parameter selection_neuron numbers
def evaluate_LSTM_model(X_train, y_train, n_neurons1,n_neurons2):
    losses = []  
    model = Sequential()
    model.add(LSTM(n_neurons1, batch_input_shape=(n_batch, X_train.shape[1], X_train.shape[2]), activation="relu",
                  return_sequences=True, stateful=True, dropout=0.2))
#model.add(Reshape((n_batch, X_train.shape[1], X_train.shape[2]), input_shape=(5,)))    
#dropout Fraction of the units to drop for the linear transformation of the inputs.
    model.add(LSTM(n_neurons2, batch_input_shape=(n_batch, X_train.shape[1], X_train.shape[2]), stateful=True))
    model.add(Dense(y_train.shape[1]))
    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(loss='mean_squared_error', optimizer="adam")
    history = model.fit(X_train, y_train, epochs=2,verbose=2, batch_size=n_batch,shuffle=False)
    losses.append(history.history['loss'])
    #val_losses.append(history.history['val_loss'][0])
    forecast = model.predict(X_test,batch_size=n_batch)
    predicted = np.reshape(forecast, (forecast.size,))
    actual =np.reshape(y_test, (y_test.size,))
    error = mean_squared_error(actual,  predicted)
    model.reset_states()
    return error

record=[]
def evaluate_models(dataset, y_train,q_values,p_values):
    dataset = dataset.astype('float32')
    best_score, best_number = float("inf"), None
    for n_neurons1 in p_values:
        for n_neurons2 in q_values:
            order= (n_neurons1,n_neurons2)
            try:
                mse = evaluate_LSTM_model(X_train, y_train, n_neurons1,n_neurons2)
                record.append((mse,order))
                if mse < best_score:
                    best_score, best_number= mse, order
                    #order_1.append(order)
                print('hiden layer neurons %s MSE=%.3f' % (order,mse))
            except:
                continue
    print('Best hiden layer neurons%s MSE=%.3f' % (best_number, best_score))

# evaluate parameters
p_values = range(1,6)
q_values = range(1,6)
import warnings
warnings.filterwarnings("ignore")
evaluate_models(X_train, y_train,q_values,p_values)



