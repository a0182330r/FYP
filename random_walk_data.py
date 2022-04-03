#!/usr/bin/env python
# coding: utf-8

# In[1]:



import generate_dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical


# In[8]:

#This function takes in parameters to creating random walks
#branched = True creates num_branched_class number of random walks which have the same lower layers
#num_each_type determines how many images are made for every class of random walk
#num_layer determines the number of lower level motifs created
#num_class,num_branched class determines the number of classes and branched classes created
def create_rw_dataset(num_each_type=70, num_class=3,num_branched_class = 1 , branched = False, num_layer=1):

    #########################################################################################################################
    #The canvas size determines the size of the lower level motifs, and the data_canvas_size determines the size of the image
    if branched == True:
    
        class_dict,data_dict = generate_dataset.branched_random_walker(number_data = num_each_type, 
                                                         number_class = num_class,num_branch_class = num_branched_class
                                                         , number_layers = num_layer ,
                                                         different_dimensions = False, canvas_size = (5,5), 
                                                         data_canvas_size = (5,5))
    elif branched == False:
    
        class_dict,data_dict = generate_dataset.generate_dataset(number_data = num_each_type, number_class = num_class
                                                         , number_layers = num_layer ,
                                                         different_dimensions = False, canvas_size = (5,5), 
                                                         data_canvas_size = (5,5))
    
    #Flatten the image
    flattened_image = data_dict[0]
    flattened_image = [pixel for pixel in flattened_image for pixel in pixel]
    flattened_image = [flattened_image,]
    
    length = len(data_dict)
    i = 1
    while ( i < length):
        newlist = data_dict[i]
        newlist = [item for items in newlist for item in items]
        flattened_image = flattened_image + [newlist,]
        i = i + 1
    
    #Convert the flattened image into a dataframe
    flattened_image = pd.DataFrame(data = flattened_image)
    
    i = 1
    #Creating the labels of the image
    labels = np.linspace(0,0,num_each_type,dtype = int)
    #First, create num_each_type number of [0] as labels for class 0, then create the same number of [1]s as label for the next class...
    #Do until all classes have labels
    if branched == False:
        while(i < num_class):
            class_label = np.linspace(i,i,num_each_type,dtype = int)
            labels = [labels,class_label]
            labels = [item for items in labels for item in items]
            i = i + 1
    elif branched == True:
        while(i < num_class+num_branched_class * 2):
            class_label = np.linspace(i,i,num_each_type,dtype = int)
            labels = [labels,class_label]
            labels = [item for items in labels for item in items]
            i = i + 1

    flattened_image['label'] = pd.DataFrame(data =labels )
    
    #The first 20 of each class are test data, afterwhich its all train data
    test_data = flattened_image.iloc[:20]
    train_data = flattened_image.iloc[20:num_each_type]
    
    #The statements inside the if/else are near identical except for the while part. branched has additional branch classes.
    if branched == False:
        i = 1
        while(i < num_class):
            test_data = pd.concat([test_data,flattened_image.iloc[(i)*num_each_type:((i)*num_each_type) + 20]]).reset_index(drop = True)
            train_data = pd.concat([train_data,flattened_image.iloc[((i)*num_each_type)+20 :(i+1)*num_each_type]]).reset_index(drop = True)
            i = i + 1
    elif branched == True:
        i = 1
        while(i < num_class + num_branched_class*2):
            test_data = pd.concat([test_data,flattened_image.iloc[(i)*num_each_type:((i)*num_each_type) + 20]]).reset_index(drop = True)
            train_data = pd.concat([train_data,flattened_image.iloc[((i)*num_each_type)+20 :(i+1)*num_each_type]]).reset_index(drop = True)
            i = i + 1
    
    train_data_shuffled = train_data.sample(frac=1).reset_index(drop=True)

    #convert train, test into x and y
    y_train = train_data_shuffled.iloc[:,-1]
    x_train = train_data_shuffled.iloc[:,:-1]


    y_test = test_data.iloc[:,-1]
    x_test = test_data.iloc[:,:-1]
    
    #Creating the labels 
    if branched == False:
        y_train = pd.DataFrame(to_categorical(y_train, num_class))
        test_labels = y_test
        y_test = pd.DataFrame(to_categorical(y_test, num_class))

    elif branched == True:
        y_train = pd.DataFrame(to_categorical(y_train, num_class+ num_branched_class*2))
        test_labels = y_test
        y_test = pd.DataFrame(to_categorical(y_test, num_class + num_branched_class*2))


    
    return x_train,y_train,x_test,y_test,test_labels,class_dict
    
    