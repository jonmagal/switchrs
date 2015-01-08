# -*- coding: utf8 -*-

'''
Created on 10/09/2014

@author: Jonathas Magalhães
'''

import graphlab
from graphlab.data_structures.sframe import SFrame
from settings import DATASET_PATH

class CF(object):
    '''
    classdocs
    '''
    sframe = None
    
    
    def read_dataset(self, dataset):
        self.sframe = SFrame.read_csv(url = dataset, delimiter = '::', header = False, 
                                      column_type_hints=[int, str, int, str, float, str, int])
        assert isinstance(self.sframe, SFrame)
        self.sframe.remove_columns(column_names = ['X2', 'X4', 'X6', ])
        print self.sframe.column_names()

    def create(self):
        m = graphlab.recommender.create(observation_data = self.sframe, user_id = 'X1', item_id = 'X3', target = 'X5')
        recs = m.recommend()
        print recs
    
class PearsonUU(CF):
    pass


if __name__ == '__main__':
    cf = CF()
    cf.read_dataset(dataset = DATASET_PATH + 'r1.train')
    cf.create()



