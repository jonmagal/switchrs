# -*- coding: utf8 -*-
'''
Created on 28/05/2014

@author: Jonathas Magalhães
'''
PATH = '/Users/jon/Dropbox/Academia/'
DATASET_PATH    = PATH + 'Datasets/'
#PATH            = '/home/seu-madruga/Dropbox/Academia/Datasets/MovieLens/'

#cf.read_dataset(dataset = DATASET_PATH + 'r1.train')
REC_MODELS_PATH     = PATH + 'Artigos_Escritos/AI_2015/analise/rec_models/'
REC_PREDICTION_PATH = PATH + 'Artigos_Escritos/AI_2015/analise/rec_predictions/'

MOVIELENS_PATH      = DATASET_PATH + 'MovieLens/ml-10M100K/' 

MOVIELENS_SWITCH_PATH = DATASET_PATH + 'MovieLens/ml-10M100K-switch/' 

DATASET_CONF = {'movielens': {
                  'folders'  : [ 
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
                        'train' : MOVIELENS_SWITCH_PATH + 'r1.train',
                        'test'  : MOVIELENS_SWITCH_PATH + 'r1.test',
                        },
    
                        {'id'   : 'r2', 
                        'train' : MOVIELENS_SWITCH_PATH + 'r2.train',
                        'test'  : MOVIELENS_SWITCH_PATH + 'r2.test',
                        },
                       
                        {'id'   : 'r3', 
                        'train' : MOVIELENS_SWITCH_PATH + 'r3.train',
                        'test'  : MOVIELENS_SWITCH_PATH + 'r3.test',
                        },
                       
                        {'id'   : 'r4', 
                        'train' : MOVIELENS_SWITCH_PATH + 'r4.train',
                        'test'  : MOVIELENS_SWITCH_PATH + 'r4.test',
                        },
                       
                        {'id'   : 'r5', 
                        'train' : MOVIELENS_PATH + 'r5.train',
                        'test'  : MOVIELENS_PATH + 'r5.test',
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