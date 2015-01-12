# -*- coding: utf8 -*-

'''
Created on 10/01/2015

@author: Jonathas Magalhães
'''
from recommender.dataset import DataSet
from recommender.rec_model import ModelManager
from switch.switch import Switch

class Evaluation():

    dataset_id      = None
    dataset         = None
    model_manager   = None
    switch          = None
    
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        self.model_manager = ModelManager()
        
    def _load_datasets(self, dataset_id = None):
        if dataset_id is not None:
            self.dataset_id = dataset_id
        if self.dataset_id is None:
            return'Error'
        else:
            self.dataset = DataSet(dataset_id = self.dataset_id, sframe = True)

    def _train_rec_models(self):
        self.model_manager.train_models(dataset = self.dataset)
        
    def _test_rec_models(self):
        self.model_manager.test_models(dataset = self.dataset)
    
    def _set_datasets_switch(self):
        self.switch = Switch()
        self.switch.prepare_dataset(dataset = self.dataset)
        
    def _train_switch(self):
        self.model_manager.test(dataset = self.dataset)
    
    def _test_switch(self):
        pass
    
    def _evaluate(self):
        pass

    def run(self, dataset_id = None):
        self._load_datasets()
        #self._train_rec_models()
        #self._test_rec_models()
        #self._set_datasets_switch()
        self._train_switch()
        
        self._test_switch()
        self._evaluate()
        
if __name__ == '__main__':
    evaluation = Evaluation(dataset_id = 'movielens')
    evaluation.run()