# -*- coding: utf8 -*-

'''
Created on 10/09/2014

@author: Jonathas Magalhães
'''

import os

from settings import REC_MODELS_PATH, REC_PREDICTION_TEST_PATH, REC_PREDICTION_TRAIN_PATH, REC_EVALUATION_PATH
from settings       import MODELS_CONF


class RecommendationModel(object):
    '''
    classdocs
    '''
    id                  = None
    model_type          = None
    options             = None
    
    def _get_model_file(self, dataset, folder):
        return REC_MODELS_PATH + self.id + '_' + dataset.id + '_' + folder.id 
        
    def _get_prediction_file(self, dataset, folder, type_prediction = 'test'):
        if type_prediction == 'test':
            return REC_PREDICTION_TEST_PATH + self.id + '_' + dataset.id + '_' + folder.id
        else:
            return REC_PREDICTION_TRAIN_PATH + self.id + '_' + dataset.id + '_' + folder.id
    
    def _get_evaluation_file(self, dataset, folder, evaluation_type = 'user'):
        if evaluation_type == 'user':
            return REC_EVALUATION_PATH + self.id + '_' + dataset.id + '_' + folder.id + '_user'
        else:
            return REC_EVALUATION_PATH + self.id + '_' + dataset.id + '_' + folder.id + '_item'
            
    def train_model(self, dataset):
        from graphlab.toolkits.recommender      import item_similarity_recommender, popularity_recommender
        from graphlab.toolkits.recommender      import factorization_recommender


        for folder in dataset.folders:
            model_file = self._get_model_file(dataset, folder) 
            
            if os.path.exists(model_file):
                print 'Recommendation Model ' + self.id + ' already trained in folder ' + folder.id + '.'
                continue
            
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
                    solver          = self.options['solver']
                    max_interations = self.options['max_iterations']
                    model = factorization_recommender.create(observation_data = folder.train_sframe, solver = solver,
                                                             target = 'rating', max_iterations = max_interations)
                
                model.save(location = model_file)
                print 'RecommendationModel ' + self.id + ' trained and saved.'
                    
    def test_model(self, dataset, type_prediction = 'test'):
        from graphlab import load_model

        for folder in dataset.folders:
            prediction_file = self._get_prediction_file(dataset, folder, type_prediction) 
            
            model_file = self._get_model_file(dataset, folder)

            if os.path.exists(prediction_file):
                print 'RecommendationModel ' + self.id + ' already tested in folder ' + folder.id + '.'
                continue 
            
            elif not os.path.exists(model_file):
                print 'Impossible testing this model. It should be trained first.'
                return
            
            else: 
                print 'Starting to test_model model ' + self.id + '.'
                model = load_model(location = model_file)
                if type_prediction == 'test':
                    predictions = model.predict(dataset = folder.test_sframe)
                else:
                    predictions = model.predict(dataset = folder.train_sframe)
                predictions.save(filename = prediction_file)
                print 'RecommendationModel ' + self.id + ' tested.'
    
    def evaluate_model(self, dataset):
        from graphlab import load_model
        
        model = None
        for folder in dataset.folders:
            model_file = self._get_model_file(dataset, folder)
        
            user_evaluation_file = self._get_evaluation_file(dataset, folder, evaluation_type = 'user')
            item_evaluation_file = self._get_evaluation_file(dataset, folder, evaluation_type = 'item')
            
            user = item = False
            
            if os.path.exists(user_evaluation_file):
                user = True
                print 'RecommendationModel ' + self.id + ' already evaluated by user in folder ' + folder.id + '.'
            
            if os.path.exists(item_evaluation_file):
                item = True
                print 'RecommendationModel ' + self.id + ' already evaluated by item in folder ' + folder.id + '.'
            
            if user and item:
                continue
            
            model       = load_model(location = model_file)
            evaluation  = model.evaluate(dataset = folder.train_sframe, metric = 'rmse')
            if not user:
                evaluation['rmse_by_user'].save(user_evaluation_file)
            if not item:
                evaluation['rmse_by_item'].save(item_evaluation_file)
            
                
    def get_prediction(self, dataset, folder, type_prediction = 'test'):
        from graphlab.data_structures.sarray    import SArray

        prediction_file = self._get_prediction_file(dataset, folder, type_prediction)
        predictions = SArray(prediction_file)
        return predictions
    
    def get_evaluation(self, dataset, folder, evaluation_type = 'item'):
        from graphlab.data_structures.sframe import SFrame
        
        evaluation_file = self._get_evaluation_file(dataset, folder, evaluation_type)
        evaluation_sframe = SFrame(evaluation_file)
        return evaluation_sframe.select_column(key = 'rmse')
        
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
            
            self.models.append(model)
    
    def train_models(self, dataset):
        for model in self.models:
            model.train_model(dataset = dataset)
            
    def test_models(self, dataset):
        for model in self.models:
            model.test_model(dataset = dataset)
            model.test_model(dataset = dataset, type_prediction = 'train')
    
    def evaluate_models(self, dataset):
        for model in self.models:
            model.evaluate_model(dataset = dataset)
            
    def get_predictions(self, dataset, folder, type_prediction = 'test'):
        predictions = [ model.get_prediction(dataset, folder, type_prediction) for model in self.models]
        return predictions
    
    def get_evaluations(self, dataset, folder, evaluation_type = 'item'):
        evaluations = [ model.get_prediction(dataset, folder, evaluation_type) for model in self.models]
        return evaluations
    
    def get_index_model(self, switch_predictions):
        return [self._get_index(model_id) for model_id in switch_predictions]
    
    def _get_index(self, model_id):
        for model in self.models:
            i = 0
            if model.id == model_id:
                return i
            i += 1 