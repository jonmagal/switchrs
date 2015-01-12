# -*- coding: utf8 -*-

'''
Created on 10/01/2014

@author: Jonathas Magalhães
'''
from settings import DATASET_CONF

class Folder(object):
    
    id                  = None
    train_file          = None
    test_file           = None
    train_sframe        = None
    test_sframe         = None
    
class DataSet(object):
    
    id      = None
    folders = []
    
    def __init__(self, dataset_id, sframe = False):
        self.id = dataset_id
        self._load_folders()
        
        self._load_sframes()
        
    def _load_folders(self):
        dataset_conf = DATASET_CONF[self.id]
        folders_conf = dataset_conf['folders']
        
        for folder_conf in folders_conf:
            folder              = Folder()
            folder.id           = folder_conf['id']
            folder.train_file   = folder_conf['train']
            folder.test_file    = folder_conf['test']
            self.folders.append(folder)
            
    def _load_sframes(self):
        from graphlab.data_structures.sframe import SFrame

        if self.id == 'movielens':
            for folder in self.folders:
                train_sframe = SFrame.read_csv(url = folder.train_file, delimiter = '::', header = False, 
                                          column_type_hints=[int, str, int, str, float, str, int])
                train_sframe.remove_columns(column_names = ['X2', 'X4', 'X6', 'X7'])
                train_sframe.rename({'X1': 'user_id', 'X3':'item_id', 'X5': 'rating'})
                folder.train_sframe = train_sframe
                
                test_sframe = SFrame.read_csv(url = folder.test_file, delimiter = '::', header = False, 
                                          column_type_hints=[int, str, int, str, float, str, int])
                test_sframe.remove_columns(column_names = ['X2', 'X4', 'X6', 'X7'])
                test_sframe.rename({'X1': 'user_id', 'X3':'item_id', 'X5': 'rating'})
                folder.test_sframe = test_sframe
                
                
class DatasetManager():
    
    def get_datasets(self):
        pass
        