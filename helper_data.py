# -*- coding: utf-8 -*-
import numpy as np
import copy
import pandas as pd
import os
import inspect

from External.notears.linear import notears_linear
import helper_dag

filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))
    
def get_data_dag(N, mode, seed_np = 42):
    
    '''
    Gaussian SCM I
    '''
    np.random.seed(seed_np)
    
    if mode == 0:

        W = np.zeros([3,3])
        sigma_sq = 1
        
        W[0,1] = 1
        W[1,2] = 2
    
    elif mode == 1:
        
        W = np.zeros((3,3))
        sigma_sq = 1
        
        W[:,1] = np.array([-.9,0,0])
        W[:,2] = np.array([-.8,0,0])

    X, S = sample_dag(W, N, 
                      sigma_sq,
                      seed = None, 
                      return_S = True)
    
    mu = np.zeros(len(W))
    data_dic = {}
    if N != 10000:
        data_dic['N'] = N
        
    pear = S/np.sqrt(np.outer(np.diag(S),np.diag(S)))
    print(np.round(pear,2))
    
    return X, mu, S, sigma_sq, W, data_dic

def sample_dag(W, N, 
               sigma_sq,
               seed = None, 
               return_S = False):
    '''
    In:
        W:      p,p
        sigma_sq:
        N:
    '''
    
    p = len(W)
    if seed is not None:
        np.random.seed(seed)
    
    Z = np.sqrt(sigma_sq)*np.random.normal(size=(N,p))
    
    A = np.linalg.inv(np.eye(p)-W.T)
    
    X = Z @ A.T    
    
    S = sigma_sq*A @ A.T
    
    if not return_S:
        return X
    else:
        return X, S
    
def get_sachs():
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    currPath = os.path.dirname(os.path.abspath(filename))
    
    df = pd.read_excel(os.path.join(currPath,'1. cd3cd28.xls'))
    
    X_0 = df.to_numpy()
    X = X_0-np.mean(X_0,0)
    
    N, p = X.shape
    S_hat = X.T@X/N
    
    mu_hat = np.zeros(p)
    
    return X, mu_hat, S_hat

def get_adv_data(X, A_a, bool_ev = False):
    
    W_a, sigma_sq_a = helper_dag.get_W_from_dag(X, A_a, bool_ev)
    
    S_a = helper_dag.get_cov(W_a, sigma_sq_a)
    
    mu_a = np.zeros(len(A_a))
    
    return mu_a, S_a, W_a, sigma_sq_a

def get_sachs_columns():
    
    columns = ['raf', 'mek', 'plc', 
               'pip2', 'pip3', 'erk', 
               'akt', 'pka', 'pkc', 
               'p38', 'jnk']
    
    return columns

def get_adv_sachs(mu, S, W = None, attack_node = 'pip2'):
    
    if W is None:
        S_a = copy.deepcopy(S)
    else:
        S_a = copy.deepcopy(W)
        
    columns = get_sachs_columns()
    
    if attack_node == 'pip2':
        edge_l = [['plc','pip2'],
                  ['plc','pip3']]
        
        node_l = ['plc','pip2','pip3']
        
    elif attack_node == 'jnk':
        
        edge_l = [['pkc','jnk'],
                  ]
        
        node_l = ['pkc','jnk','p38']
        
    for name_1, name_2 in edge_l:
        idx_r = columns.index(name_1)
        idx_c = columns.index(name_2)
    
        S_a[idx_r,idx_c] = 0
        S_a[idx_c,idx_r] = 0    
    
    mu_a = mu
    
    idx_L = [columns.index(name_i) for name_i in node_l]
    idx_adv_train = np.array(idx_L)

    return S_a, mu_a, idx_adv_train


def get_hat_sachs(X):
    
    lambda1 = 0.1
    loss_type = 'l2'

    W_est = notears_linear(X, lambda1, loss_type, 
                           max_iter=100, 
                           h_tol=1e-8, 
                           rho_max=1e+16, 
                           w_threshold=0.3)
    
    return W_est
