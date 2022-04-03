#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random_walker as rw
import copy


#Make number_data amount of data of number_class classes
def generate_dataset(number_data, number_class, number_layers = 3, different_dimensions = False, canvas_size = (9,9),varying_canvas_size = []
               , data_canvas_size = (9,9)):
    random_walker_dict = {}
    data_samples_list = []

    random_walker = rw.GridRandomWalker()
    random_walker.make_layers(number_layers , different_dimensions , canvas_size , varying_canvas_size)
    random_walker_dict[0] = random_walker
    data_samples_list = random_walker.make_data_samples(n=number_data , canvas_size = data_canvas_size)
    
    i = 1
    while(i < number_class):
        random_walker = rw.GridRandomWalker()
        random_walker.make_layers(number_layers , different_dimensions , canvas_size , varying_canvas_size)
        random_walker_dict[i] = random_walker
        data_samples_list = data_samples_list+random_walker.make_data_samples(n = number_data , canvas_size = data_canvas_size)
        i = i + 1
    return random_walker_dict,data_samples_list


def branched_random_walker(number_data=70, number_class=3,num_branch_class = 1, number_layers = 2, different_dimensions = False, 
                           canvas_size = (9,9),varying_canvas_size = []
                           , data_canvas_size = (9,9)):
    #Need to choose how many unbranched random walkers, how many branched random walkers
    random_walker_dict = {}
    data_samples_list = []

    random_walker = rw.GridRandomWalker()
    random_walker.make_layers(number_layers , different_dimensions , canvas_size , varying_canvas_size)
    random_walker_dict[0] = random_walker
    data_samples_list = random_walker.make_data_samples(n=number_data , canvas_size = data_canvas_size)
    
    i = 1
    while(i < number_class):
        random_walker = rw.GridRandomWalker()
        random_walker.make_layers(number_layers , different_dimensions , canvas_size , varying_canvas_size)
        random_walker_dict[i] = random_walker
        data_samples_list = data_samples_list+random_walker.make_data_samples(n = number_data , canvas_size = data_canvas_size)
        i = i + 1
    
    i = 1
    while(i <= num_branch_class):

        random_walker = rw.GridRandomWalker()
        random_walker.make_layers(number_layers - 1 , different_dimensions, canvas_size, varying_canvas_size)
        
        branched_random_walker = copy.deepcopy(random_walker)
        random_walker.make_image(canvas_size = canvas_size,show_graph = False)
        branched_random_walker.make_image(canvas_size = canvas_size,show_graph = False)
        random_walker_dict[i+0.1] = random_walker
        random_walker_dict[i+0.2] = branched_random_walker
        data_samples_list = data_samples_list+random_walker.make_data_samples(n = number_data , canvas_size = data_canvas_size)
        data_samples_list = data_samples_list+branched_random_walker.make_data_samples(n = number_data , canvas_size = data_canvas_size)

        i = i + 1
    
    return random_walker_dict,data_samples_list

    
############################################
