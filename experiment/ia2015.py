# -*- coding: utf8 -*-

'''
Created on 10/01/2015

@author: Jonathas Magalhães
'''
from recommender.dataset import DataSet
from recommender.rec_model import ModelManager
from switch.switch import Switch

class Evaluation():

    dataset         = None
    model_manager   = None
    switch          = None
    dataset_switch  = None
    
    def __init__(self, dataset_id, dataset_switch_id):
        self.model_manager = ModelManager()
        self.dataset = DataSet(dataset_id = dataset_id, sframe = True)
        
        self.dataset_switch = DataSet(dataset_id = dataset_switch_id, sframe = False)
        self.switch = Switch()
        
    def _train_rec_models(self):
        self.model_manager.train_models(dataset = self.dataset)
        
    def _test_rec_models(self):
        self.model_manager.test_models(dataset = self.dataset)
    
    def _create_datasets_switch(self):   
        file_save = self.dataset_switch.folders[0].train_file
        print file_save
        self.switch.prepare_dataset(dataset = self.dataset, dataset_switch = self.dataset_switch, 
                                    model_manager = self.model_manager)
        
    def _train_switch(self):
        self.switch.train(dataset_switch = self.dataset_switch)
    
    def _test_switch(self):
        pass
    
    def _evaluate(self):
        pass

    def run(self):
        #self._train_rec_models()
        #self._test_rec_models()
        #self._create_datasets_switch()
        self._train_switch()
        
        #self._test_switch()
        self._evaluate()
        
if __name__ == '__main__':
    evaluation = Evaluation(dataset_id = 'movielens', dataset_switch_id = 'movielens_switch')
    evaluation.run()