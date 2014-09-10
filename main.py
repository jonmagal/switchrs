# -*- coding: utf8 -*-
'''
Created on 28/05/2014

@author: Jonathas Magalhães
'''
from util.util import read_the_dataset
from settings import DATASET_PATH

if __name__ == '__main__':
    for x in range(1,2):
        read_the_dataset(the_dataset_file = DATASET_PATH + 'r' + str(x) + '.train')