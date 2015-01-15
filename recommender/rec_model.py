# -*- coding: utf8 -*-

'''
Created on 10/09/2014

@author: Jonathas Magalhães
'''

import graphlab
import os

from graphlab.data_structures.sframe    import SFrame
from graphlab.toolkits.recommender      import item_similarity_recommender, popularity_recommender
from graphlab.toolkits.recommender      import factorization_recommender
from graphlab.toolkits.recommender.util import compare_models

from local_settings import REC_MODELS_PATH, REC_PREDICTION_PATH
from settings       import MODELS_CONF


class RecommendationModel(object):
    '''
    classdocs
    '''
    id                  = None
    model_type          = None
    options             = None
    #model_file          = None
    #prediction_file     = None
    #evaluation_file     = None
    #parameter_file      = None
    
    def _get_model_file(self, dataset, folder):
        return REC_MODELS_PATH + self.id + '_' + dataset.id + '_' + folder.id 
        
    def _get_prediction_file(self, dataset, folder):
        return REC_PREDICTION_PATH + self.id + '_' + dataset.id + '_' + folder.id
    
    def train_model(self, dataset):
        for folder in dataset.folders:
            model_file = self._get_model_file(dataset, folder) 
            
            if os.path.exists(model_file):
                print 'Recommendation Model ' + self.id + ' already trained.'
            else:
                print 'Starting to train model ' + self.id + '.'
                
                if self.model_type == 'item_based':
                    k       = self.options['only_top_k']
                    sim     = self.options['similarity_type']
                    model   = item_similarity_recommender.create(observation_data = folder.train_sframe, 
                                                                 only_top_k = k, 
                                                                 target = 'rating', similarity_type = sim) 
                    
                elif self.model_type == 'popularity':
                    model = popularity_recommender.create(observation_data = folder.train_sframe, target = 'rating')
                    
                elif self.model_type == 'matrix_factorization':
                    solver = self.options['solver']
                    model = factorization_recommender.create(observation_data = folder.train_sframe, solver = solver,
                                                             target = 'rating')
                
                model.save(location = model_file)
                print 'RecommendationModel ' + self.id + ' trained and saved.'
                    
    def test_model(self, dataset):
        for folder in dataset.folders:
            prediction_file = self._get_prediction_file(dataset, folder) 
            model_file = self._get_model_file(dataset, folder)

            if os.path.exists(prediction_file):
                print 'RecommendationModel ' + self.id + ' already tested.'
        
            elif not os.path.exists(model_file):
                print 'Impossible testing this model. It should be trained first.'
                return
            
            else: 
                print 'Starting to test_model model ' + self.id + '.'
                model = graphlab.load_model(location = model_file)
                predictions = model.predict(dataset = folder.test_sframe)
                predictions.save(filename = prediction_file)
                print 'RecommendationModel ' + self.id + ' tested.'
    
    def get_prediction_switch(self, dataset, folder):
        model_file = self._get_model_file(dataset, folder)
        model = graphlab.load_model(location = model_file)
        predictions = model.predict(dataset = folder.train_sframe)
        return predictions
    
        
class ModelManager(object):
    
    models = []
    
    def __init__(self):
        self._set_models()
        
    def _set_models(self):
        for model_id in sorted(MODELS_CONF.keys()):
            model = RecommendationModel()
            
            model.id            = model_id
            model_conf          = MODELS_CONF[model.id]
            model.model_type    = model_conf['model_type']
            model.options       = model_conf['options']
            
            #model_obj.evaluation_file   = EVALUATION_PATH + model_obj.name + '_' + dataset.dataset_key + '_evaluation.dat'
            #model_obj.model_file        = MODEL_PATH + model_obj.name + '_' + dataset.dataset_key + '.model'
            #model_obj.prediction_file   = PREDICTION_PATH + model_obj.name + '_' + dataset.dataset_key + '_prediction.dat'
            #model_obj.parameter_file    = PARAMETER_PATH + model_obj.name + '_' + dataset.dataset_key + '_parameter.dat'
            self.models.append(model)
    
    def train_models(self, dataset):
        for model in self.models:
            model.train_model(dataset = dataset)
            
    def test_models(self, dataset):
        for model in self.models:
            model.test_model(dataset = dataset)
    
    def get_predictions_switch(self, dataset, folder):
        predictions = [ model.get_prediction_switch(dataset, folder) for model in self.models]
        return predictions