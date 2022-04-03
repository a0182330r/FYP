#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This notebook is for the curve class.
#The curve class can be quadratic, sin, cos, sin*cos
#There can be up to n parameters
#There can be a noise generated onto x
#The input parameters are (function of curve, number of parameters)



# In[28]:


import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from tensorflow.keras.utils import to_categorical


# In[29]:


class Curve():
#function specifies the function which maps inputs to outputs
    def __init__(self,function = "cosine"):
        self.function = function
    #Since the random function generates from 0 to 1, size increases the 
    #Limits of the data from 0 to size
        self.size = 20
    
#Generate Datapoints generates num_data number of data points
#The data is stored in a dataframe type
    def generate_datapoints(self, num_data):
    
        x_values = (np.random.random((num_data,1))-0.5) * self.size
        x_values = pd.DataFrame(data = x_values) 
        
        if self.function == "cosine":
            y_values = (x_values).apply( np.cos) 
        
        elif self.function == "damped_cosine":
            cosine_part = x_values.apply(np.cos)
            exponential_part = (-0.05*x_values).apply(np.exp)
            y_values = cosine_part * exponential_part
            
        elif self.function == "noisy_function":
            cosine_part = (0.5*x_values).apply(np.cos)
            sine_part = (x_values).apply(np.sin)
            y_values = cosine_part * sine_part
            
        #Noise is made here
            noise = np.random.normal(0,0.1,num_data)
            noise = pd.DataFrame(data = noise)
            y_values = y_values + noise
        
        elif self.function == "quad":
            y_values = (0.1*x_values).apply(np.square) + 1
            
    #Return, none of these values are stored within the object itself
        return x_values,y_values
    
class Mnist():
    def __init__(self):
        self.data = pd.read_csv("mnist.csv")
        
    def generate_datapoints(self,num_data):
        #Shuffles the data so a new set of datapoints is obtained every time
        data = self.data.sample(frac = 1).reset_index(drop = True)
        
        x_values = pd.DataFrame(data = data.iloc[:num_data,1:])
        #There is an additional label which is used to label the PCA plot later
        y_labels = pd.DataFrame(data = data.iloc[:num_data,0])
        y_values = pd.DataFrame(to_categorical(y_labels, 10))

        return x_values,y_values,y_labels
    
    def create_pca_datapoints(self,num_data):
        dataset = self.data[self.data["label"] == 0 ].iloc[:num_data]
        
        i = 1
        while(i < 10):
            dataset = pd.concat([dataset,self.data[self.data["label"] == i ].iloc[:num_data]])
            i = i + 1
        dataset.reset_index(drop = True)
        
        x_values = pd.DataFrame(data = dataset.iloc[:,1:])
        #There is an additional label which is used to label the PCA plot later
        y_labels = pd.DataFrame(data = dataset.iloc[:,0])
        y_values = pd.DataFrame(to_categorical(y_labels, 10))

        return x_values,y_values,y_labels
class Curve2d():
    def __init__(self):
        self.size = 20
        
    def generate_datapoints(self,num_data):
        x1_values = (np.random.random((num_data,1))-0.5) * self.size
        x2_values = (np.random.random((num_data,1))-0.5) 
        x1_values = pd.DataFrame(data = x1_values)
        x2_values = pd.DataFrame(data = x2_values)
        
        cosine_component = (x1_values ).apply(np.cos)
        sine_component = 6*(x2_values).apply(np.sin)+1
        y_values = cosine_component * sine_component
        
        x_values = pd.concat([x1_values,x2_values],axis = 1)
        return x_values,y_values
        
    
#Curve related functions : 
#Plotting it on a plot
def plot_graph(x_values,y_values,title):
    plt.plot(np.array(x_values),np.array(y_values),"bo")
    plt.xlabel("X-Inputs")
    plt.ylabel("Y-Outputs")
    plt.title(title)
    plt.show()

def train_test_split( x_values , y_values , y_labels = 0 , split = 0.8 ):
    split_point = int(split * x_values.shape[0])
    
    x_train = x_values.iloc[:split_point]
    y_train = y_values.iloc[:split_point]
    
    x_test = x_values.iloc[split_point:]
    y_test = y_values.iloc[split_point:]
    
    if(type(y_labels) == pd.DataFrame):
        #Applies only if mnist
        train_labels = y_labels.iloc[:split_point]
        test_labels = y_labels.iloc[split_point:]
        
        return x_train,y_train,x_test,y_test,train_labels,test_labels
    
    return x_train,y_train,x_test,y_test

def activation_outputs(x_values , function = "None"):
    if function == "cosine":
        y_values = (x_values).apply(np.cos) 

    elif function == "damped_cosine":
        cosine_part = x_values.apply(np.cos)
        exponential_part = (-0.05*x_values).apply(np.exp)
        y_values = cosine_part * exponential_part

    elif function == "noisy_function":
        cosine_part = (0.5*x_values).apply(np.cos)
        sine_part = (x_values).apply(np.sin)
        y_values = cosine_part * sine_part
        
    #Noise is made here
        num_data = x_values.shape[0]
        noise = np.random.normal(0,0.1,num_data)
        noise = pd.DataFrame(data = noise)
        y_values = y_values + noise

    elif function == "quad":
        y_values = (0.1*x_values).apply(np.square) + 1

    
    elif function == "curve2d":
        
        cosine_component = (x_values.iloc[:,0]).apply(np.cos)
        sine_component = 6*(x_values.iloc[:,1]).apply(np.sin) + 1
        y_values = sine_component*cosine_component

    return y_values


# In[ ]:




