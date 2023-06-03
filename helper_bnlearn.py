# -*- coding: utf-8 -*-
import pickle
import numpy as np

import helper_dag

import os
import copy
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))
	
def get_data(name):

    path = os.path.join(currPath, 'bnlearn', name+".p")
    
    with open(path, "rb") as f:
        X_0, columns, A_0 = pickle.load(f)
        
    X = X_0-np.mean(X_0, 0)
    W, sigma_sq = helper_dag.get_W_from_dag(X, A_0, bool_ev = False)
    
    columns = columns.to_list()
    
    mu = np.zeros(len(W))
    
    return X, mu, W, sigma_sq, columns

    
def get_W_a(X, W, sigma_sq, name, columns):
    
    '''
    Returns the adversarial coef matrix and nodes to mask
    '''
    W_a = copy.deepcopy(W)
    sigma_sq_a = copy.deepcopy(sigma_sq)
    
    if name == 'magic-niab':
        
        name_target = 'G1294'
        
        source = columns.index('G418')
        target = columns.index(name_target)    
        
    elif name == 'magic-irri':
        
        name_target = 'G4145'
        
        source = columns.index('G4156')
        target = columns.index(name_target)
        
    
    W_a[source,target] = 0
    
    if not np.isscalar(sigma_sq):
        sigma_sq_a[target] = np.std(X[:, target])**2
        
    idx_L = [source,target]
    idx_adv_train = np.array(idx_L)
    
    return W_a, sigma_sq_a, idx_adv_train, name_target