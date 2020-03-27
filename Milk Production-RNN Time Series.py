import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r"C:\Users\Ishan\Documents\Python Scripts\Datasets\monthly-milk-production-pounds.csv",index_col="Month")
data.head()
data.index=pd.to_datetime(data.index)
data.plot()
data.info()
train_set=data.head(156)
test_set=data.tail(12)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
trained_scaled=scaler.fit_transform(train_set)
test_scaled=scaler.fit_transform(test_set)

def next_batch(training_data,batch_size,steps):
    rand_start=np.random.randint(0,len(training_data)-steps)
    y_batch=np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)
    return y_batch[:,:-1].reshape(-1,steps,1),y_batch[:,1:].reshape(-1,steps,1)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
num_time_steps=12
num_inputs=1
num_neurons=100
num_outputs=1
learning_rate=0.03
num_train_iteration=4000
batch_size=1
X=tf.placeholder(tf.float32,[None,num_time_steps,num_inputs])
Y=tf.placeholder(tf.float32,[None,num_time_steps,num_outputs])
import tensorflow as tf1
cell=tf1.contrib.rnn.OutputProjectWrapper(tf1.contrib.rnn.GRUcell(num_units=num_neurons,activation=tf.nn.relu),output_size=num_outputs)
























































