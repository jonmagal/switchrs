# -*- coding: utf8 -*-
'''
Created on 10/01/2015

@author: Jonathas Magalhães
'''

import numpy as np
from local_settings import SWITCH_MODELS_PATH
#import numpy

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
        return SWITCH_MODELS_PATH + self.id + '_' + dataset.id + '_' + folder.id + '.pkl'
        
    def _get_prediction_file(self, dataset, folder):
        return SWITCH_MODELS_PATH + self.id + '_' + dataset.id + '_' + folder.id + '.pkl'
    
    def train(self, dataset_switch):
        from sklearn.naive_bayes    import GaussianNB
        from sklearn.externals      import joblib
        
        for folder in dataset_switch.folders:
            data_train = folder.train_file
        
            X = np.genfromtxt(fname = data_train, delimiter = ',', usecols = range(3, 29), skip_header = 1, 
                              missing_values = '', filling_values = 0)
            Y = np.genfromtxt(fname = data_train, delimiter = ',', usecols = [29], skip_header = 1)
        
            naive = GaussianNB()
            naive.fit(X, Y)
            
            file_save = self._get_model_file(dataset_switch, folder)
            
            joblib.dump(naive, file_save) 
            
    def prepare_dataset(self, dataset, dataset_switch, model_manager):  
        print "Starting to process movies."
        movie_sframe = self._process_movies(filename = dataset.movies_file)
        print "Movies processed."
        
        self._prepare_movielens(dataset, dataset_switch, model_manager, movie_sframe)
    
    def _get_best_class(self, target, *predictions):
        return np.argmin(np.absolute(np.subtract(predictions,target)))
        
    def _prepare_movielens(self, dataset, dataset_switch, model_manager, movie_sframe):
        import os
        import graphlab.aggregate as agg
        from graphlab.data_structures.sarray import SArray
        
        for folder in dataset.folders:
            
            test_file       = dataset_switch.get_folder(folder.id).test_file
            train_file      = dataset_switch.get_folder(folder.id).train_file
            
            test    = False
            train   = False
            
            
            if os.path.exists(train_file):
                print "Train file of the folder " + dataset_switch.id + " " + folder.id + " already prepared."
                train = True
            
            if os.path.exists(test_file):
                print "Test file of the folder " + dataset_switch.id + " " + folder.id + " already prepared."
                test = True
                
            #If the files were generated than we cn go to the next folder
            if train and test:
                print "Folder " + dataset_switch.id + " " + folder.id + " already prepared."
                continue
            
            print "Preparing folder " + dataset_switch.id + " " + folder.id + "."
            
            train_sframe = folder.train_sframe
            
            '''
            user_count_rating   = train_sframe.groupby(key_columns = 'user_id', operations = {'user_count_rating': agg.COUNT()})
            user_mean_rating    = train_sframe.groupby(key_columns = 'user_id', 
                                                 operations = {'user_mean_rating': agg.MEAN('rating')})
            user_sd_rating      = train_sframe.groupby(key_columns = 'user_id', 
                                                 operations = {'user_sd_rating': agg.STD('rating')})
            '''
            
            item_count_rating   = train_sframe.groupby(key_columns = 'item_id', operations = {'item_count_rating': agg.COUNT()})
            item_mean_rating    = train_sframe.groupby(key_columns = 'item_id', 
                                                 operations = {'item_mean_rating': agg.MEAN('rating')})
            item_sd_rating      = train_sframe.groupby(key_columns = 'item_id', 
                                                 operations = {'item_sd_rating': agg.STD('rating')})
            
            #user_attr = [user_count_rating, user_mean_rating, user_sd_rating]
            item_attr = [item_count_rating, item_mean_rating, item_sd_rating, movie_sframe]
            
            if not test:
                test_sframe = folder.test_sframe
                test_sframe = self._merge_sframes(test_sframe, user_attr = None, item_attr = item_attr)
                test_sframe.save(test_file, format = 'csv')
                print "Test file saved."
                
            if not train:
                train_sframe = self._merge_sframes(train_sframe, user_attr = None, item_attr = item_attr)

                predictions = model_manager.get_predictions_switch(dataset, folder)
                target      = train_sframe.select_column(key = 'rating')
            
                best_models = map(lambda t, *p: self._get_best_class(t, *p), target, *predictions)
                classes     = SArray(best_models)
            
                train_sframe.add_column(data = classes, name = 'class')
                train_sframe.save(train_file, format = 'csv')
                print "Train file saved."
     
    def _merge_sframes(self, frame, user_attr, item_attr):
        if user_attr:
            for f in user_attr:
                frame = frame.join(f, on = 'user_id', how = 'left')
        if item_attr:
            for f in item_attr:
                frame = frame.join(f, on = 'item_id', how = 'left')
        return frame
    
    def _process_movies(self, filename):
        from graphlab.data_structures.sarray import SArray
        from graphlab.data_structures.sframe import SFrame
        
        sframe = SFrame.read_csv(url = filename, delimiter = '=', header = False, 
                                          column_type_hints=[int, str, str])
       
        sframe.remove_columns(column_names = ['X2'])
        sframe.rename({'X1': 'item_id', })
        
        movies = []
        for movie in sframe:
            m = {genre:1 for genre in movie['X3'].split('|')}
            movies.append(m)
        
        sa = SArray(movies)
        sframe.add_column(sa, name = 'item_genre')
       
        sframe.remove_columns(column_names = ['X3', ])
        movie_sframe = sframe.unpack('item_genre', column_name_prefix = '')
        
        #Putting 0 where in the movies that does not have genre
        column_names = movie_sframe.column_names()
        for c in column_names:
            movie_sframe = movie_sframe.fillna(c, 0)

        return movie_sframe
"""    
p1 = SArray([1, 2, 3])
p2 = SArray([3, 4, 2])
target =  SArray([1, 3, 2])

s = Switch()
solution = map(lambda t, *p: s._get_best_class(t, *p), target, 
                                  *[p1, p2])
print solution
from graphlab.data_structures.sframe import SFrame
"""