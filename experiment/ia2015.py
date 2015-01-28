# -*- coding: utf8 -*-

'''
Created on 10/01/2015

@author: Jonathas Magalhães
'''
from recommender.dataset    import DataSet
from recommender.rec_model  import ModelManager
from switch.switch          import SwitchManager
from evaluator.evaluator    import Evaluator

class Experiment():

    dataset         = None
    dataset_switch  = None
    
    model_manager   = None
    switch_manager  = None
    
    evaluator       = None
    force           = None
    
    def __init__(self, dataset_id, dataset_switch_id, force):
        self._init_dir()
        
        self.dataset        = DataSet(dataset_id = dataset_id, sframe = True)
        self.dataset_switch = DataSet(dataset_id = dataset_switch_id, sframe = False)
        
        self.model_manager  = ModelManager()
        self.switch_manager = SwitchManager()
        self.evaluator      = Evaluator()
        
        self.force          = force
        
    def _init_dir(self):
        import os
        from settings import DIR

        for d in DIR:
            if not os.path.exists(d):
                os.makedirs(d)
                
    def _train_rec_models(self):
        self.model_manager.train_models(dataset = self.dataset)
        
    def _test_rec_models(self):
        self.model_manager.test_models(dataset = self.dataset)
    
    def _evaluate_rec_models(self):
        self.model_manager.evaluate_models(dataset = self.dataset)
        
    def _create_datasets_switch(self):   
        self.dataset_switch.prepare_switch_dataset(dataset = self.dataset, model_manager = self.model_manager, 
                                                   force = self.force)
        
    def _train_switch(self):
        self.switch_manager.train_models(dataset_switch = self.dataset_switch, force = self.force)
        
    def _test_switch(self):
        self.switch_manager.rating_prediction_switches(dataset = self.dataset, dataset_switch = self.dataset_switch, 
                         model_manager = self.model_manager, force = self.force)
    
    def _evaluate(self):
        self.evaluator.evaluate(dataset = self.dataset, dataset_switch = self.dataset_switch, 
                                model_manager = self.model_manager, switch_manager = self.switch_manager, 
                                force = self.force)

    def run(self):
        self._train_rec_models()
        self._test_rec_models()
        self._evaluate_rec_models()
        self._create_datasets_switch()
        self._train_switch()
        self._test_switch()
        self._evaluate()
        
if __name__ == '__main__':
    evaluation = Experiment(dataset_id = 'movielens', dataset_switch_id = 'movielens_switch', force = False)
    evaluation.run()