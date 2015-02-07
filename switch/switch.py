# -*- coding: utf8 -*-
'''
Created on 10/01/2015

@author: Jonathas Magalhães
'''

from settings import SWITCH_MODELS_PATH, SWITCH_CLASS_PREDICTION_PATH, SWITCH_RATING_PREDICTION_PATH, SWITCH_CONF
from rclassifiers import RCLassifier

class SwitchModel(object):
    
    id                  = None
    model_type          = None
    options             = None
            
    def _get_model_file(self, dataset, folder):
        return SWITCH_MODELS_PATH + self.id + '_' + dataset.id + '_' + folder.id + '.rds'
        
    def _get_class_prediction_file(self, dataset, folder):
        return SWITCH_CLASS_PREDICTION_PATH + self.id + '_' + dataset.id + '_' + folder.id + '.csv'

    def _get_rating_prediction_file(self, dataset, folder):
        return SWITCH_RATING_PREDICTION_PATH + self.id + '_' + dataset.id + '_' + folder.id
    
    def _get_switch_prediction(self, switch_predictions, *cf_predictions):
        return cf_predictions[switch_predictions]
    
    def _get_best_prediction(self, target, *predictions):
        import numpy as np
        
        return predictions[np.argmin(np.absolute(np.subtract(predictions, target)))]
    
    def train_switch(self, dataset_switch, force):
        import os 
        
        if self.id == 'best':
            return 
        
        for folder in dataset_switch.folders:
            model_file      = self._get_model_file(dataset_switch, folder)
            if os.path.exists(model_file) and not force:
                print 'Model ' + self.id + ' already tested in folder ' + folder.id + '.'
                continue 
        
            print 'Starting to train switch model ' + self.id + '.'
            train_file  = folder.train_file
            
            classifier = RCLassifier()
            if self.model_type == 'naive_bayes':
                classifier.naive_train(train_file, model_file)
            
            elif self.model_type == 'svm':
                classifier.svm_train(train_file, model_file)
                
    def test_switch(self, dataset_switch, force):
        import os
        
        if self.id == 'best':
            return 
        
        for folder in dataset_switch.folders:
            prediction_file = self._get_class_prediction_file(dataset_switch, folder)
            model_file      = self._get_model_file(dataset_switch, folder)
            
            if os.path.exists(prediction_file) and not force:
                print 'Model ' + self.id + ' already predicted in folder ' + folder.id + '.'
                continue 
            
            print 'Starting to test switch model ' + self.id + '.'
            test_file   = folder.test_file
            
            classifier = RCLassifier()
            if self.model_type == 'naive_bayes':
                classifier.naive_test(test_file, model_file, prediction_file)
            
            elif self.model_type == 'svm':
                classifier.svm_test(test_file, model_file)
    
    def rating_prediction_switch(self, dataset, dataset_switch, model_manager, force):
        from graphlab.data_structures.sframe import SFrame
        from graphlab.data_structures.sarray import SArray
        import os
        
        for folder in dataset.folders:
            rating_prediction_file  = self._get_rating_prediction_file(dataset_switch, folder)
            class_prediction_file   = self._get_class_prediction_file(dataset_switch, folder)
            
            if os.path.exists(rating_prediction_file) and not force:
                print "Model " + self.id + " in " + dataset_switch.id + " " + folder.id + " already tested."
                continue 
            
            cf_predictions      = model_manager.get_predictions(dataset, folder)
            
            if self.id == 'best':
                test_sframe = folder.test_sframe
                target      = test_sframe.select_column(key = 'rating')
                
                rating_predictions  = map(lambda t, *p: self._get_best_prediction(t, *p), target, *cf_predictions)
                rating_array        = SArray(rating_predictions)
                rating_array.save(filename = rating_prediction_file)
                
            else:
                
                sf                  = SFrame.read_csv(class_prediction_file, header = True, quote_char = '"', 
                                                      column_type_hints = [int, str])
                switch_predictions  = sf.select_column(key = 'x') 
                
                index_switch_predictions = model_manager.get_index_model(switch_predictions)
                
                rating_predictions  = map(lambda t, *p: self._get_switch_prediction(t, *p), 
                                          index_switch_predictions, *cf_predictions)
                
                rating_array = SArray(rating_predictions)
                rating_array.save(filename = rating_prediction_file)

    def get_prediction(self, dataset, folder, ):
        from graphlab.data_structures.sarray    import SArray

        prediction_file = self._get_rating_prediction_file(dataset, folder)
        predictions     = SArray(prediction_file)
        return predictions
    
class SwitchManager(object):
    '''
    classdocs
    '''
    
    switches = []
    
    def __init__(self):
        self._set_models()
        
    def _set_models(self):
        for model_id in sorted(SWITCH_CONF.keys()):
            switch = SwitchModel()
            switch.id            = model_id
            model_conf           = SWITCH_CONF[switch.id]
            switch.model_type    = model_conf['model_type']
            switch.options       = model_conf['options']
            self.switches.append(switch)
    
    def train_models(self, dataset_switch, force):
        for switch in self.switches:
            switch.train_switch(dataset_switch, force)
            switch.test_switch(dataset_switch, force)
        
    def rating_prediction_switches(self, dataset, dataset_switch, model_manager, force):
        for switch in self.switches:
            switch.rating_prediction_switch(dataset, dataset_switch, model_manager, force)
            
    
"""
from graphlab.data_structures.sarray import SArray

p1      = SArray([1, 2, 3])
p2      = SArray([3, 4, 2])
p3      = SArray([2, 3, 1])

target  = SArray([1, 3, 2])

classes = SArray([0, 1, 2])

s = SwitchManager()
solution = map(lambda t, *p: s._get_best_class(t, *p), target, *[p1, p2])
print solution

solution = map(lambda t, *p: s._get_switch_prediction(t, *p), classes, *[p1, p2, p3])
print solution

import os
from local_settings import DIR

for d in DIR:
    if not os.path.exists(d):
        os.makedirs(d)
train_file  = MOVIELENS_SWITCH_PATH + 'r1.train.csv'
test_file   = MOVIELENS_SWITCH_PATH + 'r1.test.csv'
model_file  = SWITCH_MODELS_PATH + 'test.rds'
n = SwitchModel()
n.train(train_file, test_file, model_file)
"""