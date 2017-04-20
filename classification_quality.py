#!/usr/bin/env python
import argparse
import json
import os.path

import errno
import numpy as np
import matplotlib.pyplot as plt

from simulator import SeriesSampler

from hmm_trading import *

def get_seed(nfolds):
    seed = np.random.randint(low=0, high=100000, size=nfolds)
    return seed
    
def contains_zero(ts):
    tmp_series = np.where(ts <= 0)[0]
    if len(tmp_series) > 0:
        return True
    return False

def generate_datasets(sampler, seed, learn_size, test_size, trend_start):

    sample = sampler.simulate(learn_size + test_size, trend_start, seed=seed)
    while contains_zero(sample["series"]):
        sample = sampler.simulate(learn_size + test_size, trend_start, 
                                  seed=get_seed(1)[0])
        
    learn = sample["series"][0:learn_size]
    test = sample["series"][learn_size:]
    return learn, test

def plot_hmm_results(series, predictions, title):
    xs = np.array(range(len(predictions)))
    ys = series[:-1]

    plt.scatter(x = xs, y = ys, c = predictions)
    plt.xlabel('t', fontsize=18)
    plt.tick_params(labelsize=13)
    plt.ylabel('series', fontsize=18)
    plt.title(title, fontsize=20)
    plt.show() 
    
def print_parameters_info(nfolds, tan_mean, tan_var, exp_lamb, var_noise):
    print("SERIES PARAMETERS:")
    print("N folds = {_nfolds}".format(_nfolds = nfolds))
    print("Mean tangent = {_tan_mean}".format(_tan_mean = tan_mean))
    print("Tangent variance = {_tan_var}".format(_tan_var = tan_var))
    print("Lambda = {_exp_lamb}".format(_exp_lamb = exp_lamb))
    print("Noise variance = {_var_noise}".format(_var_noise = var_noise))
    
def print_total_accuracy_info(learn_accuracy, test_accuracy):
    print("RESULTS:")
    print("Mean learn accuracy = {0:.4f}".format(np.mean(learn_accuracy)))
    print("Mean test accuracy = {0:.4f}".format(np.mean(test_accuracy)))
    print("SD learn accuracy = {0:.4f}".format(np.std(learn_accuracy)))
    print("SD test accuracy = {0:.4f}".format(np.std(test_accuracy)))
    
def print_hmm_info(means, covars, transmat):
    print("HMM PARAMETERS:")
    print("mu_0 = {_mu0}, sigma_0 = {_sigma0}".format(_mu0 = means[0, 0], _sigma0 = covars[0, 0,0]))
    print("mu_1 = {_mu1}, sigma_1 = {_sigma1}".format(_mu1 = means[1, 0], _sigma1 = covars[1, 0,0]))
    print("Transition matrix = ")
    print(transmat)
    
def process(nfolds, tan_mean, tan_var, exp_lamb, var_noise, 
            learn_size = 2000, test_size = 2000, trend_start = 1000, window_len = 15,
            plot = False, fit_parameter = False, print_hmm_parameters = False, verbose = True, 
            seed = None):
    if verbose:
        print_parameters_info(nfolds, tan_mean, tan_var, exp_lamb, var_noise)
    
    hmm_trader = HMM_trader(window_len = window_len)
    print("Window length = {_len}".format(_len = window_len))
    sampler = SeriesSampler(tan_mean=tan_mean,
                            tan_var=tan_var,
                            intensity=exp_lamb,
                            wiener_var=var_noise)
    if seed is None:
        seed = get_seed(nfolds)
        
    total_test_accuracy = np.zeros(shape = nfolds)
    total_learn_accuracy = np.zeros(shape = nfolds)
    
    for i in range(nfolds):
        learn, test = generate_datasets(sampler, seed[i], learn_size, test_size, trend_start)
        preprocessed_learn = preprocess(learn)
        preprocessed_test = preprocess(test)

        if fit_parameter:
            window_len = cv_fit_parameter(hmm_trader, preprocessed_learn, range(5, 50, 5))
            if verbose:
                print("Fitted window length = {_len}".format(_len = window_len))
            hmm_trader.set_parameter(window_len)
        
        hmm_trader.fit(preprocessed_learn)
        learn_predictions = hmm_trader.predict(preprocessed_learn)
        test_predictions = hmm_trader.predict(preprocessed_test)
        
        if print_hmm_parameters:
            means, covars, transmat = hmm_trader.get_model_parameters()
            print_hmm_info(means, covars, transmat)
            
        if plot:
            plot_hmm_results(learn, learn_predictions, "learn")
            plot_hmm_results(test, test_predictions, "test")
        
        total_learn_accuracy[i] = hmm_trader.score(preprocessed_learn, learn_predictions)
        total_test_accuracy[i] = hmm_trader.score(preprocessed_test, test_predictions)
    
    if verbose:
        print_total_accuracy_info(total_learn_accuracy, total_test_accuracy)
    
    return total_learn_accuracy, total_test_accuracy, window_len
