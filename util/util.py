# -*- coding: utf8 -*-
'''
Created on 09/09/2014

@author: Jonathas Magalhães
'''

import numpy as np
from settings import USERS, MOVIES

def read_the_dataset(the_dataset_file, movies):
    matrix = np.zeros((USERS, MOVIES))
    with file(the_dataset_file, 'r') as infile:
        for line in infile:    
            data = line.split('::')
            movie_index = movies.index(data[1])
            matrix[int(data[0])-1, movie_index] = float(data[2])              
    return matrix

def get_movie_vector(the_movie_file):
    movies = []
    with file(the_movie_file, 'r') as infile:
        for line in infile:    
            data = line.split('::')
            movies.append(data[0])
    return movies