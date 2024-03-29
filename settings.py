# -*- coding: utf8 -*-
'''
Created on 28/05/2014

@author: Jonathas Magalhães
'''
from local_settings import DATASET_PATH

REC_MODELS_PATH             = DATASET_PATH + 'rec_models/'
REC_PREDICTION_TEST_PATH    = DATASET_PATH + 'rec_predictions_test/' 
REC_PREDICTION_TRAIN_PATH   = DATASET_PATH + 'rec_predictions_train/'
REC_EVALUATION_PATH         = DATASET_PATH + 'rec_evaluations/'

MOVIELENS_PATH          = DATASET_PATH + 'MovieLens/ml-10M100K/'
MOVIELENS_SWITCH_PATH   = DATASET_PATH + 'MovieLens/ml-10M100K-switch/'

MOVIELENS_SWITCH_SIM_PATH   = DATASET_PATH + 'MovieLens/switch-sim/'
MOVIELENS_SWITCH_ITEM_PATH  = DATASET_PATH + 'MovieLens/switch_item/'

SWITCH_MODELS_PATH              = DATASET_PATH + 'switch_models/'
SWITCH_CLASS_PREDICTION_PATH    = DATASET_PATH + 'switch_class_predictions/'  
SWITCH_RATING_PREDICTION_PATH   = DATASET_PATH + 'switch_rating_predictions/'

RESULTS_PATH = DATASET_PATH + 'results/'
    
DIR = [DATASET_PATH, REC_MODELS_PATH, REC_PREDICTION_TEST_PATH, REC_PREDICTION_TRAIN_PATH, MOVIELENS_PATH, 
       MOVIELENS_SWITCH_PATH, MOVIELENS_SWITCH_SIM_PATH, SWITCH_MODELS_PATH, SWITCH_CLASS_PREDICTION_PATH, 
       SWITCH_RATING_PREDICTION_PATH, RESULTS_PATH]

DATASET_CONF = {'movielens': {
                  'movies':  MOVIELENS_PATH + 'movies.dat',
                  'folders': [ 
                       {'id'    : 'r1', 
                        'train' : MOVIELENS_PATH + 'r1.train',
                        'test'  : MOVIELENS_PATH + 'r1.test',
                        },
    
                        {'id'   : 'r2', 
                        'train' : MOVIELENS_PATH + 'r2.train',
                        'test'  : MOVIELENS_PATH + 'r2.test',
                        },
                       
                        {'id'   : 'r3', 
                        'train' : MOVIELENS_PATH + 'r3.train',
                        'test'  : MOVIELENS_PATH + 'r3.test',
                        },
                       
                        {'id'   : 'r4', 
                        'train' : MOVIELENS_PATH + 'r4.train',
                        'test'  : MOVIELENS_PATH + 'r4.test',
                        },
                       
                        {'id'   : 'r5', 
                        'train' : MOVIELENS_PATH + 'r5.train',
                        'test'  : MOVIELENS_PATH + 'r5.test',
                        },
                       ]
                  },
                
                'movielens_switch': {
                  'folders'  : [ 
                       {'id'    : 'r1', 
                        'train' : MOVIELENS_SWITCH_PATH + 'r1.train.csv',
                        'test'  : MOVIELENS_SWITCH_PATH + 'r1.test.csv',
                        },
    
                        {'id'   : 'r2', 
                        'train' : MOVIELENS_SWITCH_PATH + 'r2.train.csv',
                        'test'  : MOVIELENS_SWITCH_PATH + 'r2.test.csv',
                        },
                       
                        {'id'   : 'r3', 
                        'train' : MOVIELENS_SWITCH_PATH + 'r3.train.csv',
                        'test'  : MOVIELENS_SWITCH_PATH + 'r3.test.csv',
                        },
                       
                        {'id'   : 'r4', 
                        'train' : MOVIELENS_SWITCH_PATH + 'r4.train.csv',
                        'test'  : MOVIELENS_SWITCH_PATH + 'r4.test.csv',
                        },
                       
                        {'id'   : 'r5', 
                        'train' : MOVIELENS_SWITCH_PATH + 'r5.train.csv',
                        'test'  : MOVIELENS_SWITCH_PATH + 'r5.test.csv',
                        },
                       ]
                  },
                
                'movielens_switch_sim': {
                  'folders'  : [ 
                       {'id'    : 'r1', 
                        'train' : MOVIELENS_SWITCH_SIM_PATH + 'r1.train.csv',
                        'test'  : MOVIELENS_SWITCH_SIM_PATH + 'r1.test.csv',
                        },
    
                        {'id'   : 'r2', 
                        'train' : MOVIELENS_SWITCH_SIM_PATH + 'r2.train.csv',
                        'test'  : MOVIELENS_SWITCH_SIM_PATH + 'r2.test.csv',
                        },
                       
                        {'id'   : 'r3', 
                        'train' : MOVIELENS_SWITCH_SIM_PATH + 'r3.train.csv',
                        'test'  : MOVIELENS_SWITCH_SIM_PATH + 'r3.test.csv',
                        },
                       
                        {'id'   : 'r4', 
                        'train' : MOVIELENS_SWITCH_SIM_PATH + 'r4.train.csv',
                        'test'  : MOVIELENS_SWITCH_SIM_PATH + 'r4.test.csv',
                        },
                       
                        {'id'   : 'r5', 
                        'train' : MOVIELENS_SWITCH_SIM_PATH + 'r5.train.csv',
                        'test'  : MOVIELENS_SWITCH_SIM_PATH + 'r5.test.csv',
                        },
                       ]
                  },
                
                'movielens_all': {
                  'movies':  MOVIELENS_PATH + 'movies.dat',
                  'folders': [ 
                       {'id'    : 'ra', 
                        'train' : MOVIELENS_PATH + 'ra.train',
                        'test'  : MOVIELENS_PATH + 'ra.test',
                        },
    
                        {'id'   : 'rb', 
                        'train' : MOVIELENS_PATH + 'rb.train',
                        'test'  : MOVIELENS_PATH + 'rb.test',
                        },
                    ]
                  },
                
                'movielens_switch_all': {
                  'folders'  : [ 
                       {'id'    : 'ra', 
                        'train' : MOVIELENS_SWITCH_PATH + 'ra.train.csv',
                        'test'  : MOVIELENS_SWITCH_PATH + 'ra.test.csv',
                        },
    
                        {'id'   : 'rb', 
                        'train' : MOVIELENS_SWITCH_PATH + 'rb.train.csv',
                        'test'  : MOVIELENS_SWITCH_PATH + 'rb.test.csv',
                        },
                    ]  
                },
                
                'movielens_switch_all_sim': {
                  'folders'  : [ 
                       {'id'    : 'ra', 
                        'train' : MOVIELENS_SWITCH_SIM_PATH + 'ra.train.csv',
                        'test'  : MOVIELENS_SWITCH_SIM_PATH + 'ra.test.csv',
                        },
    
                        {'id'   : 'rb', 
                        'train' : MOVIELENS_SWITCH_SIM_PATH + 'rb.train.csv',
                        'test'  : MOVIELENS_SWITCH_SIM_PATH + 'rb.test.csv',
                        },
                    ]  
                },
                
                'movielens_switch_by_item': {
                  'folders'  : [ 
                       {'id'    : 'r1', 
                        'train' : MOVIELENS_SWITCH_ITEM_PATH + 'r1.train.csv',
                        'test'  : MOVIELENS_SWITCH_ITEM_PATH + 'r1.test.csv',
                        },
    
                        {'id'   : 'r2', 
                        'train' : MOVIELENS_SWITCH_ITEM_PATH + 'r2.train.csv',
                        'test'  : MOVIELENS_SWITCH_ITEM_PATH + 'r2.test.csv',
                        },
                       
                        {'id'   : 'r3', 
                        'train' : MOVIELENS_SWITCH_ITEM_PATH + 'r3.train.csv',
                        'test'  : MOVIELENS_SWITCH_ITEM_PATH + 'r3.test.csv',
                        },
                       
                        {'id'   : 'r4', 
                        'train' : MOVIELENS_SWITCH_ITEM_PATH + 'r4.train.csv',
                        'test'  : MOVIELENS_SWITCH_ITEM_PATH + 'r4.test.csv',
                        },
                       
                        {'id'   : 'r5', 
                        'train' : MOVIELENS_SWITCH_ITEM_PATH + 'r5.train.csv',
                        'test'  : MOVIELENS_SWITCH_ITEM_PATH + 'r5.test.csv',
                        },
                       ]
                  },
            }

MODELS_CONF = {'item_based1': 
                {'model_type'       : 'item_based',
                 'options'          : {'similarity_type': 'cosine', 'only_top_k': 50}, 
                 },
               
               'item_based2': 
                {'model_type'       : 'item_based',
                 'options'          : {'similarity_type': 'cosine', 'only_top_k': 100}, 
                 },
               
               'item_based3':
                {'model_type'       : 'item_based',
                 'options'          : {'similarity_type': 'cosine', 'only_top_k': 200}, 
                 },
               
               'item_based4':
               {'model_type'       : 'item_based',
                 'options'          : {'similarity_type': 'pearson', 'only_top_k': 50}, 
                 },
               
               'item_based5':
               {'model_type'       : 'item_based',
                 'options'          : {'similarity_type': 'pearson', 'only_top_k': 100}, 
                 },
               
               'item_based6':
               {'model_type'       : 'item_based',
                 'options'          : {'similarity_type': 'pearson', 'only_top_k': 200}, 
                 },
               
               'popularity1':
               {'model_type'       : 'popularity',
                 'options'          : {}, 
                 },
               
               'matrix_factorization1':
               {'model_type'       : 'matrix_factorization',
                 'options'          : {'solver': 'sgd', 'max_iterations': 50}, 
                 },
               
               'matrix_factorization2':
               {'model_type'       : 'matrix_factorization',
                 'options'          : {'solver': 'als', 'max_iterations': 50}, 
                 },
               
               'matrix_factorization3':
               {'model_type'       : 'matrix_factorization',
                 'options'          : {'solver': 'als', 'max_iterations': 100}, 
                 },
               
               'matrix_factorization4':
               {'model_type'       : 'matrix_factorization',
                 'options'          : {'solver': 'als', 'max_iterations': 100}, 
                 },
               
               } 

SWITCH_CONF = {'naive_bayes1': 
                    {'model_type'       : 'naive_bayes',
                     'options'          : {}, 
                     },
               
               'svm1': 
                    {'model_type'       : 'svm',
                     'options'          : {}, 
                     },
               
               'best': 
                    {'model_type'       : 'best',
                     'options'          : {}, 
                     },
               } 