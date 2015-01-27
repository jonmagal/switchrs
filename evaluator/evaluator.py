# -*- coding: utf8 -*-
'''
Created on 27/01/2015

@author: Jonathas Magalhães
'''
from settings import RESULTS_PATH

class Evaluator(object):
    '''
    classdocs
    '''
    
    def evaluate(self, dataset, dataset_switch, model_manager, switch_manager):
        import numpy as np
        import os 
        
        from graphlab.toolkits.evaluation import rmse
        from util.util import save_sheet
        
        values  = []
        info    = []
        title   = ['folder', ]
        
        save_file = RESULTS_PATH + dataset.id + '_' + dataset_switch.id + '.csv'
        
        if os.path.exists(save_file):
            print 'Datasets ' + dataset.id + ' and ' + dataset_switch.id + 'already evaluated.'  
            return 
        
        for model in model_manager.models:
            title.append(model.id)
        
        for switch in switch_manager.switches:
            title.append(switch.id)
            
        for folder in dataset.folders:
            values_row = []
            info_row    = []
            
            info_row.append(folder.id)
            
            test_sframe = folder.test_sframe
            targets     = test_sframe.select_column(key = 'rating')
            
            for model in model_manager.models:
                predictions = model.get_prediction(dataset, folder)
                value = rmse(targets, predictions)
                values_row.append(value)
            
            for switch in switch_manager.switches:
                switch_predictions = switch.get_prediction(dataset_switch, folder)
                value = rmse(targets, switch_predictions)
                values_row.append(value)
            
            info.append(info_row)
            values.append(values_row)
        
        v           = np.array(values)
        sum_values  = v.mean(axis = 0)
        info.append(['mean'])
        values.append(sum_values.tolist())
        
        content = [x+y for x,y in zip(info, values)]
        save_sheet(save_file, content, title)