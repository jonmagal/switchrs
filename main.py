# -*- coding: utf8 -*-
'''
Created on 28/05/2014

@author: Jonathas Magalhães
'''
from util.util import read_the_dataset, get_movie_vector
from settings import DATASET_PATH

if __name__ == '__main__':
    movies = get_movie_vector(the_movie_file = DATASET_PATH + 'movies.dat')
    for x in range(1,2):
        matrix = read_the_dataset(the_dataset_file = DATASET_PATH + 'r' + str(x) + '.train', movies = movies)
    line = matrix[71566,:]
    for x in line:
        print x