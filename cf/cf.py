'''
Created on 10/09/2014

@author: seu-madruga
'''

import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

class CF(object):
    '''
    classdocs
    '''
    matrix = None
    
    def __init__(self, params):
        '''
        Constructor
        '''
        pass
    
class PearsonUU(CF):
    pass

A = np.array(
[[0, 1, 0, 0, 1],
[0, 0, 1, 1, 1],
[1, 1, 0, 1, 0]])

dist_out = 1-pairwise_distances(A, metric = "correlation")
print dist_out