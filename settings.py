# -*- coding: utf8 -*-
'''
Created on 28/05/2014

@author: Jonathas Magalhães
'''
from local_settings import MOVIELENS_PATH, MOVIELENS_SWITCH_PATH

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
                  }
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
                 'options'          : {'solver': 'sgd'}, 
                 },
               
               'matrix_factorization2':
               {'model_type'       : 'matrix_factorization',
                 'options'          : {'solver': 'als'}, 
                 },
               } 