#!/usr/bin/env python
# coding: utf-8

# In[4]:


#This notebook plots:
#1. Training loss against Epoch
#2. Predicted output against input
#3. True output against input
#4. PCA Scree Plot
#5. PCA 2-D Component 1 against component 2 plot
#6. PCA 3-D Component 1 against component 2 against component 3 plot
#7. Heatmap of activation functions

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense , Input
import seaborn as sns

# In[ ]:


def extract_activation_values(model,x_test):

    #model_ = model
    weights = model.weights
    data_array = []

#Here the optimizer and loss is irrelevant. There is no training to be done, but it is used to ensure a working model
    N_INPUT = model.input.shape[1]
    OPTIMIZER = "RMSprop"
    LOSS = "mse"
    BATCH_SIZE = 10

    num_layers = 1
    max_layers = len(model.layers)
    
    while(num_layers < max_layers):
        
        current_weights = weights[:num_layers*2]
        new_model = Sequential()
        new_model.add(Input(shape = (N_INPUT,)))

        i = 0
        while( i < num_layers):
            num_neurons = model.weights[i*2].shape[1]

            new_model.add(Dense(num_neurons,activation = "relu"))
            i = i + 1
        new_model.compile(optimizer = OPTIMIZER , loss = LOSS)
        i = 0
        while( i < num_layers):
            new_model.layers[i].set_weights([np.array(current_weights[i*2]),np.array(current_weights[i*2 + 1])])
            i = i + 1

        #At this point the weights are updated, 
        data_array = data_array + [new_model.predict(x_test),]
        num_layers = num_layers + 1
    return data_array

def reshape_activations(model,activation_values):
    
    #The length of the model is length -1 because the final layer is the output layer (hence not counted)    
    length_model = len(model.layers) - 1
    i = 1
    activation_reshaped = activation_values[0].T
    while(i < length_model):
        activation_reshaped = np.concatenate((activation_reshaped,activation_values[i].T))
        i = i + 1
    
    #This is the total number of neurons
    length = len(activation_values) * len(activation_values[0][0])
    #This is the number of inputs
    width = len(activation_values[0])
    
    activation_reshaped = activation_reshaped.reshape(length,width)
    activation_reshaped = pd.DataFrame(data = activation_reshaped.T)
    
    return activation_reshaped

def get_activations(model, x_test):
    activation_values = extract_activation_values(model,x_test)
    reshaped_activations = reshape_activations(model,activation_values)
    return reshaped_activations


# In[14]:


def plot_train_loss(train_loss , test_loss):
    length = len(train_loss)
    x_array = np.linspace(1,length,length)
    plt.plot(x_array,train_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Train Loss")
    plt.title("Test loss is " + test_loss)
    plt.show()

def plot_curves(model,x_test,y_test):
    predicted_values = pd.DataFrame(model.predict(x_test))

    mean_R = round(100 - (abs((predicted_values-y_test).div(y_test))*100).mean()[0],2)
    plt.plot(np.array(x_test),model.predict(x_test),"bo")
    plt.ylabel("Model Prediction")
    plt.title("Predicted Output of MLP. Mean R = %s" % mean_R)
    plt.show()
    
    plt.plot(np.array(x_test),np.array(y_test),"bo")
    plt.ylabel("True Values")
    plt.title("True Values of Input")
    plt.show()

def plot_curves2d(model,x_test,y_test):
    plt.plot(np.array(x_test.iloc[:,0]),model.predict(x_test),"bo")
    plt.ylabel("Model Prediction")
    plt.title("Predicted Output of MLP")
    plt.show()
    
    plt.plot(np.array(x_test.iloc[:,0]),np.array(y_test),"bo")
    plt.ylabel("True Values")
    plt.title("True Values of Input")
    plt.show()

    
    
def plot_pca(activation_values,activation_data):
    pca = PCA(n_components = 30)
    pca_latent_dm0 = pca.fit_transform(activation_values)

    f, axes = plt.subplots(1,2, figsize=(12,4))
    axes[0].plot(pca.singular_values_, '-kx')
    #axes[0].set_yscale('log')
    axes[0].set_title("Singular value list")
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("singular value")

    axes[1].scatter(pca_latent_dm0[:,0], pca_latent_dm0[:,1],c = activation_data , s=5)
    axes[1].set_title(r"Latent space representation ($L=2$ dimensions)")
    axes[1].set_xlabel("pca component 1")
    axes[1].set_ylabel("pca component 2")
    plt.show()
    
def plot_pca_3d(activation_values,activation_data):
    pca = PCA(n_components = 30)
    pca = pca.fit_transform(activation_values)
    Xax = pca[:,0]
    Yax = pca[:,1]
    Zax = pca[:,2]
    
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')

    fig.patch.set_facecolor('white')
    length = len(Xax)
    i = 0

    ax.scatter(Xax, Yax, Zax,s=40 , c= activation_data)
    plt.show()
    
def plot_heatmap(activation_values,col_cluster = False, row_cluster = True):
    vmax = activation_values.max().max()
    sns.clustermap(activation_values.T,col_cluster=col_cluster,row_cluster = row_cluster ,vmin=0, vmax=vmax)
    plt.show()
    

def plot_image_pca(data,model):
    #Class to plot is an integer indicating which class in class dict to plot

#Flatten the image
    flat_list = []
    length = len(data)
    i = 0

    while(i < length):
        flat = data[i]
        flat = [items for items in flat for items in items]
        flat_list = flat_list + [flat,]
        i = i + 1
    flat_list = pd.DataFrame(data = flat_list)

#Get the activation values using these flat images as inputs
    activation_values = get_activations(model,flat_list)
    
#Plotting the PCA plot
    pca = PCA(n_components = 15)
    
    pca_latent_dm0 = pca.fit_transform(activation_values)
    graph_x = pca_latent_dm0[:,0]
    graph_y = pca_latent_dm0[:,1]

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    xlim_max = graph_x.max()
    ylim_max = graph_y.max()
    xlim_min = graph_x.min()
    ylim_min = graph_y.min()

#Here we are replacing pca plot's points with the image
    n = len(data)
    num_data = int(n//5)
    label = np.linspace(0,0,num_data)
    i = 1
    while(i<5):
        label = np.concatenate((label,np.linspace(i,i,num_data)))
        
        i = i + 1
    
    i = 0
    while(i< n):
        p0 = (graph_x[i]-xlim_min)/(xlim_max-xlim_min)
        p1 = (graph_y[i]-ylim_min)/(ylim_max-ylim_min)
        inset2 = fig.add_axes([p0, p1, .1, .1])
        inset2.imshow(data[i], cmap=plt.get_cmap('bone'))
        #inset2.imshow(gmm.means_[d].reshape(-1,g_sz))
        plt.setp(inset2, xticks=[], yticks=[])
        i = i + 1
    ax.axis("off")
    plt.show()
#Here we plot the original pca

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()

    ax.scatter(graph_x,graph_y, s=5,c=label)
    ax.set_title(r"Latent space representation ($L=2$ dimensions)")
    ax.set_xlabel("pca component 1")
    ax.set_ylabel("pca component 2")
    plt.show()

    
#Fraction shows the fraction of the image being lit up by the random walk * 1000.
#top, bottom, full decide which graphs to plot. top means plotting the fraction for the top half of the image, bottom for bottom half
#full takes in the whole image and finds the fraction
def plot_fraction_pca(data,model,top=True,bottom=True,full = True):
    #Class to plot is an integer indicating which class in class dict to plot
    
#Flatten the image
    flat_list = []
    length = len(data)
    i = 0

    while(i < length):
        flat = data[i]
        flat = [items for items in flat for items in items]
        flat_list = flat_list + [flat,]
        i = i + 1
    flat_list = pd.DataFrame(data = flat_list)

    center_point = flat_list.shape[1]//2+1
    top_list = flat_list.iloc[:,:center_point] #Taking the mean of only half of the data
    bottom_list = flat_list.iloc[:,center_point:] #Taking the mean of only half of the data
    fractions_full = np.around(np.array((flat_list*1000).T.mean()),2)
    fractions_top = np.around(np.array((top_list*1000).T.mean()),2)
    fractions_bottom = np.around(np.array((bottom_list*1000).T.mean()),2)
    
#Get the activation values using these flat images as inputs
    activation_values = get_activations(model,flat_list)
    
#Plotting the PCA plot
    pca = PCA(n_components = 15)
    
    pca_latent_dm0 = pca.fit_transform(activation_values)
    graph_x = pca_latent_dm0[:,0]
    graph_y = pca_latent_dm0[:,1]
    
    if full == True:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot()

        ax.scatter(graph_x,graph_y, s=5)
        k = 0
        for i , j in zip(graph_x,graph_y):
            ax.annotate(str(fractions_full[k]),xy=(i,j))
            k = k + 1
        ax.set_title("Fraction of image covered by random walk")
        ax.set_xlabel("pca component 1")
        ax.set_ylabel("pca component 2")
        plt.show()

    if top == True:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot()

        ax.scatter(graph_x,graph_y, s=5)
        k = 0
        for i , j in zip(graph_x,graph_y):
            ax.annotate(str(fractions_top[k]),xy=(i,j))
            k = k + 1
        ax.set_title("Fraction of top half of image covered by random walk")
        ax.set_xlabel("pca component 1")
        ax.set_ylabel("pca component 2")
        plt.show()

    if bottom == True:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot()

        ax.scatter(graph_x,graph_y, s=5)
        k = 0
        for i , j in zip(graph_x,graph_y):
            ax.annotate(str(fractions_bottom[k]),xy=(i,j))
            k = k + 1
        ax.set_title("Fraction of bottom half of image covered by random walk")
        ax.set_xlabel("pca component 1")
        ax.set_ylabel("pca component 2")
        plt.show()
