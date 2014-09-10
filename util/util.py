# -*- coding: utf8 -*-
'''
Created on 09/09/2014

@author: Jonathas Magalhães
'''

import numpy as np
from settings import USERS, MOVIES

def read_the_dataset(the_dataset_file):
    matrix = np.zeros((USERS, MOVIES))
    
    dict_movie = {}
    value = 0
    with file(the_dataset_file,'r') as infile:
        for line in infile:    
            data = line.strip('::')
            if data[1] in dict_movie:
                movie_index = dict_movie.get(data[1]) 
            else:
                dict_movie[data[1]] = value
                value +=1 
            matrix[int(data[0]), movie_index] = float(data[2])  
            
    return matrix