#!/usr/bin/env python
# coding: utf-8

# In[15]:


#This file contains the code for the building and testing of a MLP model

import numpy as np
import random
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense , Input
from keras.initializers import RandomNormal



# In[16]:


def build_model_curves(num_layers = 4 , num_neurons = 50 , input_shape = 1 , optimizer = "rmsprop" ):
    loss = "mse"
    
    
    model = Sequential()
    model.add(Input(shape = (input_shape,)))
    i = 0 
    #A layer with the given number of neurons is created, and this process is repeated for however many layers there are
    while(i < num_layers):
        model.add(Dense( num_neurons , activation = "relu" ,kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.039, seed = None) ))
        i = i + 1
    #This is the output layer
    model.add(Dense(1))
    
    model.compile(optimizer = optimizer, loss = loss)
    
    return model

#The input/output specifications in here are for MNIST dataset
def build_model_images(num_layers = 4 , num_neurons = 50 , input_shape = 784 , output_shape = 10 , optimizer = "rmsprop" ):

    model = Sequential()
    model.add(Input(shape=(input_shape,)))

    i = 0 
    while(i < num_layers):
        model.add(Dense( num_neurons, activation = "relu",  kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.039, seed = None)))
        i = i + 1
        
#This is the output layer. Unlike the Curve model, it uses softmax + cce as the numerical distance between types is meaningless
#ie the difference between type 1 and type 2 is not necessarily smaller than the difference between type 1 and type 3
    model.add(Dense(output_shape, activation = "softmax"))
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy" , metrics = ["accuracy"])
    
    return model

def build_model_curve2d(num_layers = 4 , num_neurons = 50 , optimizer = "rmsprop" ):
    loss = "mse"
    
    
    model = Sequential()
    model.add(Input(shape = (2,)))
    i = 0 
    #A layer with the given number of neurons is created, and this process is repeated for however many layers there are
    while(i < num_layers):
        model.add(Dense( num_neurons , activation = "relu" ,kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.039, seed = None) ))
        i = i + 1
    #This is the output layer
    model.add(Dense(1))
    
    model.compile(optimizer = optimizer, loss = loss)
    
    return model


# In[19]:


def train_model(model , x_train , y_train , x_test , y_test , epoch = 30):
    
    history = model.fit(
        x_train,
        y_train,
        batch_size = 20,
        epochs = epoch
    )
#The training and test loss are obtained. The train loss can be plotted to show the training has finished (reached a plateau)
#The test loss is useful for checking for overtraining
    train_loss = history.history["loss"]
    test_loss = model.evaluate(x_test,y_test)
    return model, train_loss, test_loss


# In[ ]:




