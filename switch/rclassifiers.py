'''
Created on 27/01/2015

@author: seu-madruga
'''

class RCLassifier(object):
    '''
    classdocs
    '''


    def naive_train(self, train_file, model_file):
        from rpy2 import robjects
        
        robjects.r('''
                    train <- function(train_file, model_file) {
                        library(e1071)
                        train         <- read.csv(train_file)
                        train$class   <- as.factor(train$class)
                        naive         <- naiveBayes(train[, 4:26], train[, 27])
                        saveRDS(naive, model_file)
                    }
                    ''')
        
        train = robjects.globalenv['train']
        train(train_file, model_file)
        
    def naive_test(self, test_file, model_file, prediction_file):
        from rpy2 import robjects
      
        robjects.r('''
                    pred <- function(test_file, model_file, prediction_file) {
                        library(e1071)
                        naive         <- readRDS(model_file)
                        test          <- read.csv(test_file)
                        pred          <- predict(naive, test[, 4:26])
                        write.csv(pred, prediction_file)
                    }
                    ''')
        pred = robjects.globalenv['pred']
        pred(test_file, model_file, prediction_file)