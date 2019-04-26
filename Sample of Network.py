# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:34:30 2019

@author: alexliuyi

Sample Code for Keras

"""

from keras.datasets import mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

#Preparing Training Data
train_data = train_data.reshape((60000,28*28))
train_data = train_data.astype('float32')/255

test_data = test_data.reshape((10000,28*28))
test_data = test_data.astype('float32')/255

#Preparing Labels
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)



#%%
#Structure Models
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
#%%
#Fit Model
network.fit(train_data,train_labels,epochs=5, batch_size=128)

#%%
#Test Model
test_loss, test_acc = network.evaluate(test_data,test_labels)
print('test_acc: ', test_acc)
#%%