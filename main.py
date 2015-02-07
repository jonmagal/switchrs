# -*- coding: utf8 -*-
'''
Created on 28/05/2014

@author: Jonathas Magalhães
'''

import sys, getopt
from experiment.ia2015 import Experiment

def main(argv):

    dataset     = 'movielens'
    switch      = 'movielens_switch'
    force       = False
    
    try:
        opts, args = getopt.getopt(argv,"d:s:f:",["dataset=","switch=", "force=", ])
    except getopt.GetoptError:
        print 'test.py -d <dataset> -s <switch> -f <force>'
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-s", "--switch"):
            switch = arg
        elif opt in ("-f", "--force"):
            force = True
            
    return dataset, switch, force
            
if __name__ == '__main__':
    dataset, switch, force = main(sys.argv[1:])
    evaluation = Experiment(dataset_id = dataset, dataset_switch_id = switch, force = force)
    evaluation.run()