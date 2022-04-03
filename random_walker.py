#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import numpy as np 
import random
from collections import OrderedDict
import matplotlib.pyplot as plt



class GridRandomWalker():
    
        def __init__(self, canvas_size=(9,9)):
        
            if (canvas_size[0]%2 != 1 or canvas_size[1]%2 != 1):
                print("Canvas size must be odd")
                sys.exit(0)

            self.canvas_size = canvas_size
            self.rad_x = np.floor(canvas_size[0]/2)
            self.rad_y = np.floor(canvas_size[1]/2)
            #rad and canvas_size are kept as self due to being used in functions
            
            #There are the elementary shapes used to make the first image

            o = np.float32(0) #0
            l = np.float32(1) #1
            type0 = [[o,o,o],[o,o,o],[o,o,o]]
            type1 = [[o,o,o],[l,l,l],[o,o,o]]
            type2 = [[o,l,o],[o,l,o],[o,l,o]]
            type3 = [[o,l,o],[l,l,o],[o,o,o]]
            type4 = [[o,l,o],[o,l,l],[o,o,o]]
            type5 = [[o,o,o],[l,l,o],[o,l,o]]
            type6 = [[o,o,o],[o,l,l],[o,l,o]]
            layer0 = { 0:type0, 1: type1 , 2 : type2 , 3 : type3 
                           , 4 : type4 , 5 : type5 , 6 : type6}
            #o and l are used instead of 0 and 1 to define the dtype
            
            
            #image_dict captures the motif of every layer.
            self.image_dict = {}
            self.image_dict[0] = layer0
            #Counter is used to toggle image dict. it increases by 1 everytime a motif is saved in image dict to prevent overwriting.
            self.counter = 1
            
    ########################################################
    # make_trajectory method takes in n (0 or 1) and makes a trajectory of type n. Makes use of _clip_and_merge to make a single continuous trajectory
        def make_trajectory(self, n):
            if not self.has_no_valid_moves[n]:

                last_point = list(self.trajectory[n].keys())[-1]
                np.random.shuffle(self.rand_neighbour_indices)

                new_point = None
                for neighbour_index in self.rand_neighbour_indices:
                    neighbour = self._tuple_return(neighbour_index,last_point)
                    if self._check_boundary(neighbour) and not neighbour in self.trajectory[n]:
                        new_point = neighbour 
                        break

                if new_point is None: 
                    self.has_no_valid_moves[n] = True
                else:
                    self.trajectory[n][new_point] = self.count[n]
                    self.count[n] += 1

                    if new_point in self.trajectory[n-1]:
                        merged_trajectory = self._clip_and_merge(self.trajectory[0],self.trajectory[1],new_point,n)
                        self.merged_trajectory = merged_trajectory
                        self.merged = True

                        
    #################################################################    
    #Takes the trajectories made in make_trajectory, saves 
        def _attempt_random_walk (self, template_type):
            
            if template_type == 0:
                self.start_point = tuple([0,self.rad_y])
                self.end_point = tuple([self.rad_x , 2*self.rad_y])
                self.point = [self.start_point,self.end_point]
            elif template_type == 1:
                self.start_point = tuple([0,self.rad_y])
                self.end_point = tuple([2*self.rad_x,self.rad_y])
                self.point = [self.start_point,self.end_point]

            self.trajectory = [OrderedDict(),OrderedDict()]
            self.trajectory[0][self.start_point] = 0
            self.trajectory[1][self.end_point] = 0

            self.has_no_valid_moves = [False,False]
           
            self.count = [1,1]
            self.rand_neighbour_indices = np.arange(4)
                
            self.merged = False
            while (self.merged == False):

                self.make_trajectory(0)

                self.make_trajectory(1)

                if self.has_no_valid_moves[0] and self.has_no_valid_moves[1]:
                    return [False, None]
            return [True,self.merged_trajectory]
                        
    #########################################################################
        def _tuple_return(self, neighbor_index, p):
            #p is the current coordinates, and neighbor_index is a randomly chosen integer. this function picks a direction for the random walker
            if neighbor_index==0:
                return tuple([p[0],p[1]-1])
            if neighbor_index==1:
                return tuple([p[0]-1,p[1]])
            if neighbor_index==2:
                return tuple([p[0],p[1]+1])
            if neighbor_index==3:
                return tuple([p[0]+1,p[1]])

###########################################################################
#Checks if the new coordinate obtained from _tuple_return lies in the canvas. 
        def _check_boundary(self, x):
            return  0 <= x[0] < self.canvas_size[0] and 0 <= x[1] < self.canvas_size[1]
        
        
###########################################################################
#When 2 trajectories meet, clip away the excess from one trajectory (so that it forms a continous line instead of branches)
#Then, merge reverses the order of trajectory 2 and combines it into one trajectory.
        def _clip_and_merge(self, trajectory1, trajectory2, intersection_point, trajectory_to_clip):
            #Which trajectory to clip is decided by whether its the head of trajectory1 that meets the body of trajectory0, or
            #if its the head of trajectory0 that meets the body of trajectory1
            if trajectory_to_clip == 0:
                L1 = [tuple(point) for point in trajectory1]
                if trajectory2[self.point[1]] < trajectory2[intersection_point]:
                    # add in reverse order
                    L2 = [tuple(point) for point in trajectory2 if trajectory2[point] < trajectory2[intersection_point]]
                    L2.reverse()
                    L = L1 + L2
                  
                elif trajectory2[self.point[1]] == trajectory2[intersection_point]:
                    L = L1
                    
                else:
                    assert False
                    
            if trajectory_to_clip == 1:
                L1 = [tuple(point) for point in trajectory2]
                if trajectory1[self.point[0]] < trajectory1[intersection_point]:
                    # add in reverse order
                    L2 = [tuple(point) for point in trajectory1 if trajectory1[point] < trajectory1[intersection_point]]
                    L2.reverse()
                    L = L1 + L2
                    L.reverse()
                   
                elif trajectory1[self.point[0]] == trajectory1[intersection_point]:
                    L = L1
                    L.reverse()
                    
                else:
                    assert False

            return L
                
#################################################################                
#The problem with attempt_random_walk is that the trajectories may never meet, as the trajectories meet dead ends. 
#Instead of forcing them to meet somehow, create_random_walks repeats the process until attempt_random_walk successfully creates a usable trajectory
        def create_random_walks(self):
            type0_is_done = False
            type1_is_done = False
            while not type0_is_done:
                [type0_is_done, self.type0_trajectory] = self._attempt_random_walk(0)
            while not type1_is_done:
                [type1_is_done , self.type1_trajectory] = self._attempt_random_walk(1)
                
                
###################################################################
#The next section converts the trajectory into an array (of 1 and 0's) representing an image.

#A nxn matrix is made to represent each pixel of the random walk, where each element of the matrix is a value from 0 to 6.
#A value of 0 means the walk did not touch that pixel, hence that pixel is just black. For non-0 values, each integer is linked to a specific motif based
#on the direction of the path of the walker.

#Then, the image is made by converting each element in the matrix above into its corresponding motif, then joining all the motifs together piecewise.

###################################################################
#Given the trajectory, we now know what the previous and next vectors are, from which we can deduce what the start and end of the motif should be.
#There are 6 motifs (L shapes and straight, along with the various rotations), and decide number picks the motif to use based on the start and end.

#Here, it is important to note that there is no differentiation between start and end, ie the motif of an image that goes from right to left looks 
#identical to a motif that starts from left to right. Will this be a problem?
        def decide_number(self,start,end):
            if start == "Left" and  end == "Right":
                number = 1
            elif start == "Right" and end == "Left":
                number = 1
            elif start == "Up" and end == "Bottom":
                number = 2
            elif start == "Bottom" and end == "Up":
                number = 2
            elif start == "Left" and end == "Bottom":
                number = 3
            elif start == "Bottom" and end == "Left":
                number = 3
            elif start == "Right" and end == "Bottom":
                number = 4
            elif start == "Bottom" and end == "Right":
                number = 4
            elif start == "Up" and end == "Left":
                number = 5
            elif start == "Left" and end == "Up":
                number = 5
            elif start == "Up" and end == "Right":
                number = 6
            elif start == "Right" and end == "Up":
                number = 6
            else:
                number = 7 #7 is the warning number which is not associated with anything. If it comes here it means an error will occur.
            return number
        
##########################################################################
# makes the matrix representing each pixel of the random walk
        def make_matrix(self,trajectory,n):
            array = np.zeros(self.canvas_size,dtype = np.float32)
            
            start = "Left" #Start is always left in type0 and type1
            is_horizontal = trajectory[1][0] - trajectory[0][0]
            is_vertical = trajectory[1][1] - trajectory[0][1]
            if is_horizontal == 1:
                end = "Right"
                #We skip left because start is alr left, left-left doesnt occur
            elif is_vertical == 1:
                end = "Up"
            elif is_vertical == -1:
                end = "Bottom"
            
            number = self.decide_number(start,end)
            array[int(trajectory[0][1])][int(trajectory[0][0])] = number
            
            #The last coordinate is excluded as it does not have a "end"
            length = len(trajectory)-1
            i = 1
            while(i < length):
                is_horizontal = trajectory[i][0]-trajectory[i-1][0]
                is_vertical = trajectory[i][1] - trajectory[i-1][1]
                if is_horizontal == 1 :
                    start = "Left"
                elif is_horizontal == -1:
                    start = "Right"
                elif is_vertical == 1 :
                    start = "Bottom"
                elif is_vertical == -1:
                    start = "Up"

                is_horizontal = trajectory[i+1][0]-trajectory[i][0]
                is_vertical = trajectory[i+1][1] - trajectory[i][1]
                if is_horizontal == 1 :
                    end = "Right"
                elif is_horizontal == -1:
                    end = "Left"
                elif is_vertical == 1 :
                    end = "Up"
                elif is_vertical == -1:
                    end = "Bottom"

                number = self.decide_number(start,end)
                array[int(trajectory[i][1])][int(trajectory[i][0])] = number
                i = i + 1
            
            #Depending on whether it is type 0 or type 1 the end changes
            if n == 0:
                end = "Up"
            elif n == 1:
                end = "Right"
            is_horizontal = trajectory[-1][0]-trajectory[-2][0]
            is_vertical = trajectory[-1][1] - trajectory[-2][1]
            if is_horizontal == 1 :
                start = "Left"
            elif is_horizontal == -1 :
                start = "Right"
            elif is_vertical == 1:
                start = "Bottom"
            elif is_vertical == -1:
                start = "Up"
            number = self.decide_number(start,end)
            array[int(trajectory[-1][1])][int(trajectory[-1][0])] = number
            
            return array
                
##############################################################
# Converts each element of the matrix into a motif
        def turn_matrix_to_image(self,matrix):
        #The motifs are stored in the previous layer of image_dict.
            image = self.image_dict[self.counter-1][matrix[0][0]]
            length = len(matrix[0])
            i = 1
            while(i < length):
                #The concat function joins each motif piece by piece horizontally
                image = np.concatenate((image,self.image_dict[self.counter-1][matrix[0][i]]),axis = 1)
                i = i + 1
            #Here image is the first row of the motifs
            
            #row is first made which contains a row of motifs, then it is joined to image vertically
            width = len(matrix)
            j = 1
            while(j < width):
                row = self.image_dict[self.counter-1][matrix[j][0]]
                i = 1 
                while(i < length):
    
                    row = np.concatenate((row,self.image_dict[self.counter-1][matrix[j][i]]),axis = 1)
                    i = i + 1
            
                image = np.concatenate((image,row),axis = 0)
                j = j + 1
            
            
            return image

#################################################################
#This function just combines make matrix and turn matrix to image so that the user can make an image with a single command.
        def make_image(self , show_graph = True,canvas_size = (9,9)):
            
            #The choice of canvas size will change the size of the next layer made, but does not affect already made layers.
            self.canvas_size = canvas_size
            self.rad_x = np.floor(canvas_size[0]/2)
            self.rad_y = np.floor(canvas_size[1]/2)
            self.create_random_walks()
            
            self.array0 = self.make_matrix(self.type0_trajectory,0)
            self.array1 = self.make_matrix(self.type1_trajectory,1)
            self.image0 = self.turn_matrix_to_image(self.array0)
            self.image1 = self.turn_matrix_to_image(self.array1)
              
            if show_graph == True:
                plt.subplot(1, 2, 1)
                plt.imshow(self.image0, cmap=plt.get_cmap('gray'))
                plt.title("Type 0 Trajectory")

                plt.subplot(1,2 , 2)
                plt.imshow(self.image1, cmap=plt.get_cmap('gray'))
                plt.title("Type 1 Trajectory")
            
            self.make_variants(self.image0,self.image1)
            #Uses image1 and image0 to make the rotational variants

            
        def make_data_layer(self,canvas_size = (9,9)):
            #Makes a data image that does not create a new file in image_dict
            self.canvas_size = canvas_size
            self.rad_x = np.floor(canvas_size[0]/2)
            self.rad_y = np.floor(canvas_size[1]/2)
            self.create_random_walks()
            k = random.randint(0,1)
            
            if k == 1:
                self.array = self.make_matrix(self.type1_trajectory,1)
            elif k == 0:
                self.array = self.make_matrix(self.type0_trajectory,0)
            self.image = self.turn_matrix_to_image(self.array)
            
            return self.image
            
            
##############################################################
#The rotational modes are made by numpys rotation and transposition.
        def make_variants(self , type0 , type1):
            dictionary = {}
            dictionary[0] = np.zeros(type0.shape,dtype = np.float32)
            dictionary[1] = type1
            dictionary[2] = type1.T
            dictionary[3] = np.rot90(type0.T)
            dictionary[4] = type0.T
            dictionary[5] = type0
            dictionary[6] = np.rot90(type0)
            
            #Saves all rotations into dictionary.
            self.image_dict[self.counter] = dictionary
            self.counter = self.counter + 1
        
############################################################
#This function shows the type0 and type1 motif of said layer        
        def show_image(self,layer):
                plt.subplot(1, 2, 1)
                plt.imshow(self.image_dict[layer][5], cmap=plt.get_cmap('gray'))
                plt.title("Type 0 Trajectory")

                plt.subplot(1,2 , 2)
                plt.imshow(self.image_dict[layer][1], cmap=plt.get_cmap('gray'))
                plt.title("Type 1 Trajectory")
        
###############################################################
        def make_layers(self, number_layers, different_dimensions = False , canvas_size = (9,9) , varying_canvas_size = []):
            i = 0
            if different_dimensions == False: #All dimensions are equal

                while (i < number_layers):
                    self.make_image(show_graph = False, canvas_size = canvas_size)
                    i = i + 1

            elif different_dimensions == True: #The dimensions are different
                if len(varying_canvas_size) != number_layers:
                    print("Number of layers does not match number of canvas size specified!")
                else: #Number matches
                    while (i < number_layers):
                        self.make_image(show_graph = False, canvas_size = varying_canvas_size[i])
                        i = i + 1
                        
        def show_array(self,layer, image_type = 0):
            if image_type == 0:
                return self.image_dict[layer][5]
            elif image_type == 1:
                return self.image_dict[layer][1]
            else:
                print("Image Type is not 0 or 1!")

#################################################################
        def make_data_samples(self,n, canvas_size = (9,9)):
            i = 1
            data_list = [self.make_data_layer(canvas_size = canvas_size),]
            while(i < n):
                data_list = data_list + [self.make_data_layer(canvas_size = canvas_size),]
                i = i + 1
            return data_list

#################################################################
#To use the function:
#random_walker = GridRandomWalker()                   <<<<<<<<<     Initiates the random_walker, it has no layers yet
#random_walker.make_image(canvas_size)                <<<<<<<<<     Creates layer 1 
#random_walker.make_next_layer(canvas_size)           <<<<<<<<<     Creates next layer
#random_walker.show_image(layer)                      <<<<<<<<<     Shows the motif of the nth layer of the random walk
        
