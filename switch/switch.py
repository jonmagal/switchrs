# -*- coding: utf8 -*-
'''
Created on 10/01/2015

@author: Jonathas Magalhães
'''

from settings import SWITCH_MODELS_PATH, SWITCH_CLASS_PREDICTION_PATH, SWITCH_RATING_PREDICTION_PATH, RESULTS_FILE,\
    MOVIELENS_SWITCH_PATH

class NaiveBayes(object):
    
    def train(self, train_file, test_file, model_file):
        from rpy2 import robjects
        import os 
        
        if os.path.exists(model_file):
            print "Model already tested."  
            return 
        
        robjects.r('''
                    train <- function(train_file, test_file, model_file) {
                        library(e1071)
                        train         <- read.csv(train_file)
                        train$class   <- as.factor(train$class)
                        naive         <- naiveBayes(train[, 4:26], train[, 27])
                        saveRDS(naive, model_file)
                    }
                    ''')
        
        train = robjects.globalenv['train']
        train(train_file, test_file, model_file)
    
    def predict(self, test_file, model_file, prediction_file):
        from rpy2 import robjects
        import os
        
        if os.path.exists(prediction_file):
            print "Model already predicted."
            return 
        
        robjects.r('''
                    pred <- function(test_file, model_file, prediction_file) {
                        library(e1071)
                        naive         <- readRDS(model_file)
                        test          <- read.csv(test_file)
                        pred          <- predict(naive, test[, 4:26])
                        write.csv(pred, prediction_file)
                    }
                    ''')
        pred = robjects.globalenv['pred']
        pred(test_file, model_file, prediction_file)
        
class Switch(object):
    '''
    classdocs
    '''
    
    id                  = 'naive_bayes1'
    model_type          = 'naive_bayes'
    options             = None
    
    def __init__(self):
        '''
        Constructor
        '''
        pass
    
    def _get_model_file(self, dataset, folder):
        return SWITCH_MODELS_PATH + self.id + '_' + dataset.id + '_' + folder.id + '.rds'
        
    def _get_class_prediction_file(self, dataset, folder):
        return SWITCH_CLASS_PREDICTION_PATH + self.id + '_' + dataset.id + '_' + folder.id + '.csv'

    def _get_rating_prediction_file(self, dataset, folder):
        return SWITCH_RATING_PREDICTION_PATH + self.id + '_' + dataset.id + '_' + folder.id
    
    def _get_switch_prediction(self, switch_predictions, *cf_predictions):
        return cf_predictions[switch_predictions]
    
    def evaluate(self, dataset, dataset_switch, model_manager):
        import numpy as np
        
        from graphlab.data_structures.sarray import SArray
        from graphlab.toolkits.evaluation import rmse
        from util.util import save_sheet
        
        values  = []
        info    = []
        title = ['folder', ]
        
        for model in model_manager.models:
            title.append(model.id)
        title.append(self.id)
            
        for folder in dataset.folders:
            values_row = []
            info_row    = []
            
            info_row.append(folder.id)
            
            test_sframe = folder.test_sframe
            targets     = test_sframe.select_column(key = 'rating')
            
            rating_prediction_file  = self._get_rating_prediction_file(dataset_switch, folder)
            switch_predictions = SArray(rating_prediction_file)
            
            for model in model_manager.models:
                predictions = model.get_prediction(dataset, folder)
                value = rmse(targets, predictions)
                values_row.append(value)
                
            value = rmse(targets, switch_predictions)
            values_row.append(value)
            
            info.append(info_row)
            values.append(values_row)
        
        v           = np.array(values)
        sum_values  = v.mean(axis = 0)
        info.append(['mean'])
        values.append(sum_values.tolist())
        
        content = [x+y for x,y in zip(info, values)]
        save_sheet(RESULTS_FILE, content, title)
        
    
    def test(self, dataset, dataset_switch, model_manager):
        from graphlab.data_structures.sframe import SFrame
        from graphlab.data_structures.sarray import SArray
        import os
        
        for folder in dataset.folders:
            rating_prediction_file  = self._get_rating_prediction_file(dataset_switch, folder)
            class_prediction_file   = self._get_class_prediction_file(dataset_switch, folder)
            
            if os.path.exists(rating_prediction_file):
                print "Model " + self.id + " in " + dataset_switch.id + " " + folder.id + " already tested."
                continue 
            
            cf_predictions      = model_manager.get_predictions(dataset, folder)
            
            
            sf                  = SFrame.read_csv(class_prediction_file, header = True, quote_char = '"', 
                                                  column_type_hints=[int, str])
            switch_predictions  = sf.select_column(key = 'x') 
            
            index_switch_predictions = model_manager.get_index_model(switch_predictions)
            
            rating_predictions  = map(lambda t, *p: self._get_switch_prediction(t, *p), 
                                      index_switch_predictions, *cf_predictions)
            
            rating_array = SArray(rating_predictions)
            rating_array.save(filename = rating_prediction_file)
    
    def train(self, dataset_switch):
        
        naive = NaiveBayes()
        for folder in dataset_switch.folders:
            train_file  = folder.train_file
            test_file   = folder.test_file
            
            prediction_file = self._get_class_prediction_file(dataset_switch, folder)
            model_file      = self._get_model_file(dataset_switch, folder)
            
            naive.train(train_file, test_file, model_file)
            naive.predict(test_file, model_file, prediction_file)
    

"""
from graphlab.data_structures.sarray import SArray

p1      = SArray([1, 2, 3])
p2      = SArray([3, 4, 2])
p3      = SArray([2, 3, 1])

target  = SArray([1, 3, 2])

classes = SArray([0, 1, 2])

s = Switch()
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
n = NaiveBayes()
n.train(train_file, test_file, model_file)
"""