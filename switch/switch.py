# -*- coding: utf8 -*-
'''
Created on 10/01/2015

@author: Jonathas Magalhães
'''
import graphlab.aggregate as agg
from graphlab.data_structures.sarray import SArray
import numpy as np
#import numpy

class Switch(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        pass
    
    def train(self, dataset_switch):
        from sklearn.naive_bayes import GaussianNB
        
        folder = dataset_switch.folders[0]
        data_train = folder.train_file
        
        X = np.genfromtxt(fname = data_train, delimiter = ',', usecols = range(3, 29), skip_header = 1, 
                          missing_values = '', filling_values = 0)
        Y = np.genfromtxt(fname = data_train, delimiter = ',', usecols = [29], ski_header = 1)
        
        naive = GaussianNB()
        naive.fit(X, Y)
        
    def prepare_dataset(self, dataset, dataset_switch, model_manager):  
        self._prepare_movielens(dataset, dataset_switch, model_manager)
    
    def get_best_class(self, target, *predictions):
        return np.argmin(np.absolute(np.subtract(predictions,target)))
        
    def _prepare_movielens(self, dataset, dataset_switch, model_manager):
        #for folder in dataset.folders:
        folder = dataset.folders[0]
        sframe = folder.train_sframe
        
        user_count_rating   = sframe.groupby(key_columns = 'user_id', operations = {'user_count_rating': agg.COUNT()})
        user_mean_rating    = sframe.groupby(key_columns = 'user_id', 
                                             operations = {'user_mean_rating': agg.MEAN('rating')})
        user_sd_rating      = sframe.groupby(key_columns = 'user_id', 
                                             operations = {'user_sd_rating': agg.STD('rating')})
        
        item_count_rating   = sframe.groupby(key_columns = 'item_id', operations = {'item_count_rating': agg.COUNT()})
        item_mean_rating    = sframe.groupby(key_columns = 'item_id', 
                                             operations = {'item_mean_rating': agg.MEAN('rating')})
        item_sd_rating      = sframe.groupby(key_columns = 'item_id', 
                                             operations = {'item_sd_rating': agg.STD('rating')})
        
        movie_sframe = self._process_movies(filename = dataset.movies_file)
        
        #for array in user_count:
        #    print array
        
        s1 = sframe.join(user_count_rating, on = 'user_id', how = 'left')
        s2 = s1.join(user_mean_rating, on = 'user_id', how = 'left')
        s3 = s2.join(user_sd_rating, on = 'user_id', how = 'left')
        s4 = s3.join(item_count_rating, on = 'item_id', how = 'left')
        s5 = s4.join(item_mean_rating, on = 'item_id', how = 'left')
        s6 = s5.join(item_sd_rating, on = 'item_id', how = 'left')
        s7 = s6.join(movie_sframe, on = 'item_id', how = 'left')
        
        #{'rating': 4.0, 'Sci-Fi': None, 'Crime': None, 'Romance': 1, 'item_id': 1393, 'Animation': None, 'Comedy': None, 'War': None, 'user_id': 14623, 'user_sd_rating': 0.8084393211002503, 'Fantasy': None, 'Horror': None, 'Film-Noir': None, 'Musical': None, 'Adventure': None, 'Thriller': None, 'Western': None, 'Mystery': None, 'item_sd_rating': 0.9265691984755607, 'Drama': 1, 'IMAX': None, 'Action': None, '(no genres listed)': None, 'Documentary': None, 'user_mean_rating': 3.3995983935743, 'user_count_rating': 498, 'item_count_rating': 10097, 'item_mean_rating': 3.6352381895612607, 'Children': None}
        
        predictions = model_manager.get_predictions_switch(dataset, folder)
        target = sframe.select_column(key = 'rating')
        
        best_models = map(lambda t, *p: self.get_best_class(t, *p), target, *predictions)
        classes = SArray(best_models)
        s7.add_column(data = classes, name = 'class')
        
        file_save = dataset_switch.folders[0].train_file
        
        s7.save(file_save, format = 'csv')
        
    def _process_movies(self, filename):
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
        
        return movie_sframe
"""    
p1 = SArray([1, 2, 3])
p2 = SArray([3, 4, 2])
target =  SArray([1, 3, 2])

s = Switch()
solution = map(lambda t, *p: s.get_best_class(t, *p), target, 
                                  *[p1, p2])
print solution
from graphlab.data_structures.sframe import SFrame
"""