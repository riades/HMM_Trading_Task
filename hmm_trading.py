#!/usr/bin/env python

import pandas as pd
import numpy as np
from hmmlearn import hmm
import warnings
    
def preprocess(ts):
    ts_diff = np.diff(ts)
    return ts_diff

def cv_fit_parameter(hmm_trader, series, grid, nfolds = 5):
    series_folds = np.array_split(series, nfolds)
    acc = np.zeros_like(grid, dtype = np.float)
    
    for i in range(len(acc)):
        hmm_trader.set_parameter(grid[i])
        
        for k in range(nfolds):
            series_train = list(series_folds)
            series_test  = series_train.pop(k)
            series_train = np.concatenate(series_train)
            hmm_trader.fit(series_train)
            predictions = hmm_trader.predict(series_test)
            acc[i] += hmm_trader.score(series_test, predictions)
        
        acc[i] /= nfolds
    return grid[np.argmax(acc)]

class HMM_trader():
    
    def __init__(self, window_len):
        self.hmm_model = None
        self.correct = True
        self.window_len = window_len
        
    def set_parameter(self, window_len):
        self.window_len = window_len
        
    def correct_labels(self):
        means = self.hmm_model.means_
        self.correct = means[0][0] < 0 and means[1][0] > 0

    def fit(self, learn_series):
        hmm_model = hmm.GaussianHMM(n_components = 2, covariance_type = "diag", 
                                    n_iter = 100)
        hmm_model.fit(pd.DataFrame(learn_series))
        self.hmm_model = hmm_model
        self.correct_labels()    
        
    def get_model_parameters(self):
        means = self.hmm_model.means_
        covars = self.hmm_model.covars_
        transmat = self.hmm_model.transmat_

        if not self.correct:
            means = means[::-1]
            covars = covars[::-1,::-1]
            transmat = transmat[::-1,::-1]
        return means, covars, transmat
        
    def predict(self, test_series):
        warnings.filterwarnings("ignore", category = DeprecationWarning)
        
        predictions = np.zeros_like(test_series)
        for i in range(1, len(test_series)):
            w = test_series[max(0, i - self.window_len) : i]
            predictions[i] = self.hmm_model.predict(pd.DataFrame(w))[-1]
        
        if self.correct:
            return predictions
        else:
            return 1 - predictions
    
    def score(self, series, predictions):
        series_sign = (np.sign(series) + 1) / 2
        confusion_matrix = np.zeros(shape = (2, 2), dtype = np.int)

        for (i, j) in zip(series_sign, predictions):
            confusion_matrix[int(i)][int(j)] += 1

        accuracy = float(confusion_matrix[0][0] + confusion_matrix[1][1]) / len(predictions)
        return accuracy 

    
