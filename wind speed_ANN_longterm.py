# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 17:20:00 2017
https://gist.github.com/HasseIona/4bcaf9f95ae828e056d5210a2ea07f88
https://medium.com/@williamkoehrsen/deep-neural-network-classifier-32c12ff46b6c
learning rate： 太大，震荡不收敛，太小，收敛慢。

@author: fei
"""
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error
rcParams['figure.figsize'] = 1, 1
#from sklearn.preprocessing import LabelEncoder
from pandas import concat
from numpy import ndarray
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
from matplotlib import pyplot
from numpy import array
import numpy as np
import seaborn as sns
from scipy import stats, integrate
from sklearn.metrics import mean_absolute_error


df=pd.read_csv("Boston_Wind_Dataset.csv")
df=df.replace(0,1)
time_step=1
batch_size = 1
n_test=8760
n_val=8760

n_hidden_1=5
n_hidden_2=5
n_hidden_3=10
n_hidden_4=10
learning_rate=0.005
training_epochs=10000


#Feature scaling
#scaler1 = StandardScaler()
scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler2 = MinMaxScaler(feature_range=(0, 1))
scale_X =df.loc[:,["Daily_data","Hourly_data","Monthly_data","Pre_year_data"]]
scale_Y =df.loc[:,["Label_year_data"]]
scalerX = scaler.fit(scale_X)
scalery = scaler.fit(scale_Y)

scaled_X = scalerX.transform(scale_X)
scaled_X = DataFrame(scaled_X)
scaled_X.columns=["Daily_data","Hourly_data","Monthly_data","Pre_year_data"]
scaled_Y = scalery.transform(scale_Y)
scaled_Y = DataFrame(scaled_Y)
scaled_Y.columns=["Label_year_data"]

###adding time sig and cos
x = scaled_X.join(df.loc[:,["Hour","Month"]])
def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data
#x = encode(x, 'Hour', 23)
x=encode(x,"Month",12)
x=x.drop(["Hour"],axis=1)
x=x.drop(["Month"],axis=1)
#train_x, test_x ,X_val= x[:(len(scaled_X)-n_val-n_test)], x[-n_test:], x[n_val:(n_test+n_val)]

train_x, test_x ,X_val= scaled_X[:(len(scaled_X)-n_val-n_test)], scaled_X[-n_test:], scaled_X[n_val:(n_test+n_val)]
train_y, test_y,y_val = scaled_Y[:(len(scaled_X)-n_val-n_test)], scaled_Y[-n_test:],scaled_Y[n_val:(n_test+n_val)]


print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(y_val.shape)
train_x=train_x.astype('float32')
test_x=test_x.astype('float32')
train_y=train_y.astype('float32')
test_y=test_y.astype('float32')
X_val=X_val.astype("float32")
y_val=y_val.astype("float32")

#define the important parameters and variable to work with the tensorflow
cost_history=np.empty(shape=[1],dtype=float)
n_dim=train_x.shape[1]
n_dimy=train_y.shape[1]
print("n_dim",n_dim)
print("n_dimy",n_dimy)

#feed in 多组数据 用nona
x = tf.placeholder(tf.float32, [None, n_dim])
y_= tf.placeholder(tf.float32, [None,n_dimy])

#feedin batch_size，
#x = tf.placeholder(tf.float32, [batch_size, n_dim],name='x_placeholder')
#y_= tf.placeholder(tf.float32, [batch_size,n_seq])

def multilayer(x,weights,biases):
   #hidden layer
    layer_1=tf.add(tf.matmul(x, weights["h1"]), biases["b1"]) 
    layer_1=tf.nn.relu(layer_1)
    
    layer_2=tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"]) 
    layer_2=tf.nn.relu(layer_2)
    
#    layer_3=tf.add(tf.matmul(layer_2, weights["h3"]), biases["b3"]) 
#    layer_3=tf.nn.relu(layer_3)
    
#    layer_4=tf.add(tf.matmul(layer_3, weights["h4"]), biases["b4"]) 
#    layer_4=tf.nn.relu(layer_4)
    
    #outputlayer
    out_layer=tf.matmul(layer_2,weights["out"])+biases["out"]
    return out_layer

#define the weights and the biases for each layer      
weights={
     "h1":tf.Variable(tf.truncated_normal([n_dim,n_hidden_1])),
     "h2":tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
#     "h3":tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
#     "h4":tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_4])),
     "out":tf.Variable(tf.truncated_normal([n_hidden_2,n_dimy]))     
            }  

biases={
     "b1":tf.Variable(tf.truncated_normal([n_hidden_1])),
     "b2":tf.Variable(tf.truncated_normal([n_hidden_2])),
#     "b3":tf.Variable(tf.truncated_normal([n_hidden_3])),
#     "b4":tf.Variable(tf.truncated_normal([n_hidden_4])),
     "out":tf.Variable(tf.truncated_normal([n_dimy]))              
          }  
 
#call your model defined
y = multilayer(x,weights,biases)
#define the cost function and optimizer
#cost_function=tf.reduce_mean(tf.reduce_sum(tf.square(y-y_),reduction_indices=1))
cost_function  = tf.reduce_mean(tf.squared_difference(y_, y))
#cost_function  = tf.losses.mean_squared_error(y_, y)
training_step=tf.train.GradientDescentOptimizer(learning_rate). minimize(cost_function) 
#training_step=tf.train.AdamOptimizer(). minimize(cost_function) 


#initialize all  the variables 
init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)  


Accuracy_history=[]
saver=tf.train.Saver()
cost_history=[]
for i in range(training_epochs):
    start = (i*batch_size) % (len(train_x))
    end = start+batch_size
    sess.run(training_step,feed_dict={x:train_x[start:end],y_:train_y[start:end]}) 
#    sess.run(training_step,feed_dict={x:train_x,y_:train_y})
    cost=sess.run(cost_function,feed_dict={x:train_x,y_:train_y})
#    cost_history=np.append(cost_history,cost)
    cost_history.append(cost)
#    make a prediction  
    Accuracy=sess.run(cost_function,feed_dict={x:X_val,y_:y_val})
    Accuracy_history.append(Accuracy)
    
#    i += 1
    print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(cost))
    print("Epoch:", '%04d' % (i+1), "validation=", "{:.9f}".format(Accuracy))

w_value,b_value=sess.run([weights,biases])
# fiish the training model

performance = pd.DataFrame({'Training' :cost_history,
                                'Validation' : Accuracy_history}, 
            columns=['Training','Validation'])

#performance.to_csv('performance_hidden2_long.csv',index=False)


pyplot.figure(1, figsize = (10,6))
pyplot.plot(cost_history, label='MSE_Train',c="r")
pyplot.plot(Accuracy_history, label='MSE_validation',c="b")
pyplot.title('model train vs validation loss')
#pyplot.ylim(ymax = 0.9, ymin = 0.01)
#pyplot.xlim(xmax = 50, xmin = 1)
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

pyplot.title('model train loss')
pyplot.plot(cost_history[0:50], label='MSE_Train',c="r")
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.show()

# make a prediction
pred_y=sess.run(y,feed_dict={x:test_x})
# invert scaling for forecast
inv_yhat = scalery.inverse_transform(pred_y)
real_test = scale_Y[17520:26280]
from scipy import stats
print(stats.describe(inv_yhat))
print(stats.describe(real_test))

# calculate MSE with scaled 
for i in range(n_dimy):
    rmse = sqrt(mean_squared_error(test_y, pred_y))   
    mse = mean_squared_error(test_y, pred_y)
    MAE=mean_absolute_error(test_y, pred_y)
#    MAPE=mean_absolute_percentage_error(test_y, pred_y)
    print('t+%d RMSE: %f' % ((i+1), rmse)) 
    print('t+%d MAE: %f' % ((i+1), MAE)) 
    print('t+%d MSE: %f' % ((i+1), mse))

#with inversed value
for i in range(n_dimy):
    rmse = sqrt(mean_squared_error(real_test, inv_yhat))   
    mse = mean_squared_error(real_test, inv_yhat)
    MAE=mean_absolute_error(real_test, inv_yhat)
#    MAPE=MAE/real_test
    print('t+%d RMSE: %f' % ((i+1), rmse))   
    print('t+%d MSE: %f' % ((i+1), mse))
    print('t+%d MAE: %f' % ((i+1), MAE))



predicted=DataFrame(inv_yhat)
predicted.reset_index(drop=True, inplace=True)
predicted.columns=["Pred"]
real_test=DataFrame(real_test)
real_test.reset_index(drop=True,inplace=True)
new=real_test.join(predicted)
df3 = new[new.index % 20 == 0]
import random
#d4=df3.apply(lambda s: s[1]+random.randrange(-5,5), axis=1)
#df3.to_csv("Boston_2015_real_predicted data.csv")

pyplot.plot('Pred', label='MSE_Train',c="b",linewidth=1,linestyle='-',data=df3)
pyplot.plot('Label_year_data', label='MSE_Train',linewidth=1,linestyle=':',c="r",data=df3)
pyplot.legend(['Forecasted', 'Measured'], loc='upper right')
#pyplot.title('model train loss')
#pyplot.ylabel('Wind Speed m/s')
#pyplot.xlabel('Hours')
pyplot.grid(True,linestyle=':')
pyplot.show()

sess.close()  
###########################33333

def huber_loss(labels,predictions,delta=1.0):
    residual=tf.abs(predictions-labels)
    condition=tf.less(residual, delta)
    small_res=0.5*tf.square(residual)
    large_res=delta*residual-0.5*sqrt(delta)
    print(tf.where(condition,small_res,large_res))

a=huber_loss(test_y,pred_y,delta=1.0)
total_loss=[]
Accuracy_history=[]
with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init) 
    #print weight and bias before training
    print("weight",sess.run(weights))
    
    steps=100
    for i in range(steps):
        start = (i*batch_size) % (len(train_x))
        end = start+batch_size
        sess.run(training_step,feed_dict={x:train_x[start:end],y_:train_y[start:end]})
#        sess.run(training_step,feed_dict={x:X_val[start:end],y_:y_val[start:end]})
        if i % 5 == 0:
            cost_history=sess.run(cost_function,feed_dict={x:train_x,y_:train_y})
            total_loss.append(cost_history)
            Accuracy=sess.run(cost_function,feed_dict={x:X_val,y_:y_val})
            Accuracy_history.append(Accuracy)
            print("after %d training step;loss on all data is %g" %(i,cost_history))
            print("after %d valudiation step;loss on all data is %g" %(i,Accuracy_history))
pyplot.plot(total_loss, label='MSE_Train',c="r")
pyplot.plot(Accuracy_history, label='MSE_Train',c="b")
pyplot.show()
# after training the weight and bias values.
    print("weight",sess.run(weights))




#####################################################################
#x1 = np.array(predicted)
#x1 = np.extract(predicted==0,x1)
#np.array(predicted)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mean_absolute_percentage_error(actual, predicted)



from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPClassifier

gs = GridSearchCV(nn, param_grid={
    'learning_rate': [0.05, 0.01, 0.005, 0.001],
    'hidden0__units': [4, 8, 12,20],
    'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]})
gs.fit(X, y)

#param_grid = dict(neurons=neurons)
grid_search = GridSearchCV(estimator=multilayer(x,weights,biases), param_grid=param_grid, n_jobs=-1)
grid_result = grid_search.fit(train_x, train_y)












# How to decide weather add (,reduction_indices=[1]) or not.
# the softmax is implemented internally in tl.cost.cross_entropy(y, y_, 'cost') to
# speed up computation, so we use identity here.
# see tf.nn.sparse_softmax_cross_entropy_with_logits()

# define cost function and metric.
#y = network.outputs
#cost = tl.cost.cross_entropy(y, y_, 'cost')
#correct_prediction = tf.equal(tf.argmax(y, 1), y_)
#acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#y_op = tf.argmax(tf.nn.softmax(y), 1)
 #https://gist.github.com/HasseIona/4bcaf9f95ae828e056d5210a2ea07f88



 # calculate MSE
#mse_=tf.reduce_mean(tf.square(pred_y-test_y))  
#print('Test MSE: %.3f' % sess.run(mse_)) 
#print('RSS: %.4f'% sum((pred_y-test_y)**2))

#test_x = test_x.reshape((test_x.shape[0],1, test_x.shape[2]))
# #invert scaling for forecast
#inv_yhat = concatenate((yhat, test[:, :-1]), axis=1)
#inv_yhat = scaler.inverse_transform(inv_yhat)
#inv_yhat = inv_yhat[:,0]

#rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
#mse_= mean_squared_error(inv_y, inv_yhat)
#print('Test RMSE: %.3f' % rmse)
#print('RSS: %.4f'% sum((inv_yhat-inv_y)**2))   

#temp = train_y.shape
#train_y = train_y.reshape(temp[0], 1)
#train_y = np.concatenate((1-train_y, train_y), axis=1)
#temp2 = test_y.shape
#test_y = test_y.reshape(temp2[0], 1)
#test_y = np.concatenate((1-test_y, test_y), axis=1)