# -*- coding: utf8 -*-

'''
Created on 10/01/2014

@author: Jonathas Magalhães
'''
from settings import DATASET_CONF

class Folder(object):
    
    id                  = None
    train_file          = None
    test_file           = None
    train_sframe        = None
    test_sframe         = None
    
class DataSet(object):
    
    id      = None
    folders = None
    
    def __init__(self, dataset_id, sframe = False):
        print "Initiating the dataset " + dataset_id + "."
        self.id = dataset_id
        self._load_folders()
        
        if sframe == True:
            print "Loading dataframes of the dataset " + dataset_id + "."
            self._load_sframes()
    
    #Public methods
    def get_folder(self, folder_id):
        for f in self.folders:
            if f.id == folder_id:
                return f
                
    def prepare_switch_dataset(self, dataset, model_manager, force):  
        print "Starting to process movies."
        movie_sframe = self._process_movies(filename = dataset.movies_file)
        print "Movies processed."
        
        all_data    = False
        by_item     = False
        by_user     = False
        sim_data    = False
        
        if self.id == 'best':
            return 
        
        if self.id == 'movielens_switch_by_item':
            by_item = True
        
        if self.id == 'movielens_switch_all':
            all_data = True
        
        if self.id == 'movielens_switch_all_sim':
            all_data = True
            sim_data = True
        
        if self.id == 'movielens_switch_sim':
            sim_data = True
                
        self._prepare_movielens(dataset, model_manager, movie_sframe, by_item, by_user, all_data, sim_data, force)
        
    
    ###################################################################################################################
    
    #Private methods     
    
    def _prepare_movielens(self, dataset, model_manager, movie_sframe, by_item = False, by_user = False, 
                           all_data = False, sim_data = False, force = False):
        import os
        import graphlab.aggregate as agg
        
        for folder in dataset.folders:
            
            test_file       = self.get_folder(folder.id).test_file
            train_file      = self.get_folder(folder.id).train_file
            
            test    = False
            train   = False
            
            if os.path.exists(train_file) and not force:
                print "Train file of the folder " + self.id + " " + folder.id + " already prepared."
                train = True
            
            if os.path.exists(test_file) and not force:
                print "Test file of the folder " + self.id + " " + folder.id + " already prepared."
                test = True
                
            #If the files were generated than we cn go to the next folder
            if train and test:
                print "Folder " + self.id + " " + folder.id + " already prepared."
                continue
            
            print "Preparing folder " + self.id + " " + folder.id + "."
            
            train_sframe    = folder.train_sframe
            user_attr       = None 
            
            if all_data:
                user_count_rating   = train_sframe.groupby(key_columns = 'user_id', 
                                                            operations = {'user_count_rating': agg.COUNT()})
                user_mean_rating    = train_sframe.groupby(key_columns = 'user_id', 
                                                            operations = {'user_mean_rating': agg.MEAN('rating')})
                user_sd_rating      = train_sframe.groupby(key_columns = 'user_id', 
                                                            operations = {'user_sd_rating': agg.STD('rating')})
                user_attr           = [user_count_rating, user_mean_rating, user_sd_rating]
            
            item_count_rating   = train_sframe.groupby(key_columns = 'item_id', 
                                                       operations = {'item_count_rating': agg.COUNT()})
            item_mean_rating    = train_sframe.groupby(key_columns = 'item_id', 
                                                 operations = {'item_mean_rating': agg.MEAN('rating')})
            item_sd_rating      = train_sframe.groupby(key_columns = 'item_id', 
                                                 operations = {'item_sd_rating': agg.STD('rating')})
            item_attr = [item_count_rating, item_mean_rating, item_sd_rating, movie_sframe]
            
            if sim_data:
                similar_items_cosine    = model_manager.get_similar_items(similarity_type = 'cosine', dataset = dataset, 
                                                                          folder = folder)
                similar_items_pearson   = model_manager.get_similar_items(similarity_type = 'pearson', 
                                                                          dataset = dataset, folder = folder)
            
            if not test:
                if by_item:
                    test_sframe = item_attr.pop()
                    test_sframe = self._merge_sframes(test_sframe, user_attr = user_attr, item_attr = item_attr)
                    evaluations = model_manager.get_evaluations(dataset, folder, evaluation_type = 'item')
                    test_sframe = self._add_best_classes_rmse(test_sframe, evaluations, model_manager)
                    test_sframe.save(test_file, format = 'csv')
                    print "Test file saved."
                    
                else:
                    test_sframe     = folder.test_sframe
                    test_sframe     = self._merge_sframes(test_sframe, user_attr = user_attr, item_attr = item_attr)
                    predictions     = model_manager.get_predictions(dataset, folder, type_prediction = 'test')
                    test_sframe     = self._add_best_classes(test_sframe, predictions, model_manager)
                    test_sframe.save(test_file, format = 'csv')
                    print "Test file saved."
                
            if not train:
                if by_item:
                    pass
                else:
                    train_sframe    = self._merge_sframes(train_sframe, user_attr = user_attr, item_attr = item_attr)
                    predictions     = model_manager.get_predictions(dataset, folder, type_prediction = 'train')
                    train_sframe    = self._add_best_classes(train_sframe, predictions, model_manager)
                    train_sframe.save(train_file, format = 'csv')
                    print "Train file saved."
    
        
    def _get_best_class_rmse(self, *predictions):
        import numpy as np
        
        return np.argmin(predictions)
    
    def _add_best_classes_rmse(self, frame, evaluations, model_manager):
        from graphlab.data_structures.sarray import SArray
        
        best_models_index   = map(lambda *p: self._get_best_class_rmse(*p), *evaluations)
        best_models         = [model_manager.models[x].id for x in best_models_index]
        classes             = SArray(best_models)
        frame.add_column(data = classes, name = 'class')
        return frame
    
    def _get_best_class(self, target, *predictions):
        import numpy as np
        
        return np.argmin(np.absolute(np.subtract(predictions, target)))
    
    def _add_best_classes(self, frame, predictions, model_manager):
        from graphlab.data_structures.sarray import SArray
        
        target = frame.select_column(key = 'rating')
        best_models_index   = map(lambda t, *p: self._get_best_class(t, *p), target, *predictions)
        best_models         = [model_manager.models[x].id for x in best_models_index]
        classes             = SArray(best_models)
        frame.add_column(data = classes, name = 'class')
        return frame
        
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
      
    def _load_folders(self):
        self.folders = []
        dataset_conf = DATASET_CONF[self.id]
        folders_conf = dataset_conf['folders']
        
        for folder_conf in folders_conf:
            folder              = Folder()
            folder.id           = folder_conf['id']
            folder.train_file   = folder_conf['train']
            folder.test_file    = folder_conf['test']
            self.folders.append(folder)
            
        if self.id == 'movielens' or self.id == 'movielens_all':  
            self.movies_file = dataset_conf['movies']
            
    def _load_sframes(self):
        from graphlab.data_structures.sframe import SFrame

        if self.id == 'movielens' or self.id == 'movielens_all':
            for folder in self.folders:
                train_sframe = SFrame.read_csv(url = folder.train_file, delimiter = '::', header = False, 
                                          column_type_hints=[int, str, int, str, float, str, int])
                train_sframe.remove_columns(column_names = ['X2', 'X4', 'X6', 'X7'])
                train_sframe.rename({'X1': 'user_id', 'X3':'item_id', 'X5': 'rating'})
                folder.train_sframe = train_sframe
                
                test_sframe = SFrame.read_csv(url = folder.test_file, delimiter = '::', header = False, 
                                          column_type_hints=[int, str, int, str, float, str, int])
                test_sframe.remove_columns(column_names = ['X2', 'X4', 'X6', 'X7'])
                test_sframe.rename({'X1': 'user_id', 'X3':'item_id', 'X5': 'rating'})
                folder.test_sframe = test_sframe
                    
class DatasetManager():
    
    def get_datasets(self):
        pass
        