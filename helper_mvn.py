#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import scipy
from scipy.stats import multivariate_normal

def get_trig(S, mode = 'svd'):
    
    if mode == 'svd':
        U, s, V = scipy.linalg.svd(S,lapack_driver='gesvd')
            
        L = U @ np.diag(np.sqrt(s))
        
    else:
        L = scipy.linalg.cholesky(S)
    
    return L

def sample(mu, S, seed = None):
    
    '''
    In:     
        mu:     N,p
        S:      p,p
    
    Inter:
        Z:      N,p
    '''
    if seed is not None:
        np.random.seed(seed)
    
    Z = np.random.normal(size=mu.shape)
    L = get_trig(S, mode = 'svd')
    
    X = Z @ L.T + mu
    
    return X
    
def get_cond_prob(X_o, idx_1, idx_2, mu, S):
                 
    '''
    +
    Calculates the conditional mean and variance of the hidden R.V's given the observed R.V's.
    Input:
        X_o:    N,p2
        mu:     p,
        S:      p,p
        idx_1:  p1,
        idx_2:  p2,
        
    Inter:
        A1:     p1,p2
        A2:     N,p2
        A3:     p1,N
        
    Output:
        mu_1_2: N,p1,
        S_1_2:  p1,p1
        
    '''    
    x_2 = X_o
    
    mu_2 = mu[idx_2]
    mu_1 = mu[idx_1]
    
    #Calculating the sub cov. matrices
    
    S_11 = S[idx_1][:,idx_1]
    S_12 = S[idx_1][:,idx_2]
    S_21 = S_12.T#getSubS( S, idx_2, idx_1)
    S_22 = S[idx_2][:,idx_2]
    
    S_22_inv = scipy.linalg.inv( S_22)                  #It's used twice.
    
    A1 = S_12 @ S_22_inv                                #D_1xD_1 * p2xp2
    A2 = x_2 - mu_2
    A3 = A1 @ A2.T
    
    mu_1_2 = A3.T + mu_1
    
    S_1_2 = S_11 - A1 @ S_21                        #D_1xD_1    
    
    return mu_1_2, S_1_2

    
def get_ratio_ideal(mu_p, cov_p, 
                    mu_q, cov_q, 
                    X):
    
    '''
    +
    p/q
    
    In:
        X:    N,D
        
    Out:
        ratio:    N,
        
    '''        
    like_p = scipy.stats.multivariate_normal.pdf( X, mu_p, cov_p)
    like_q = scipy.stats.multivariate_normal.pdf( X, mu_q, cov_q)
    
    ratio = like_p / like_q        
        
    return ratio

def get_entropy(S):
    
    '''
    +
    '''
    p = S.shape[0]
    
    c = p*(np.log(2*np.pi)+1)
    
    entropy = 0.5*(c + np.linalg.slogdet(S)[1])

    return entropy

def get_KL(mu, S, mu2, S2, K2=None):
    
    '''
    +    
    https://statproofbook.github.io/P/mvn-kl.html
    Eq. (11)
    '''
    p = len(mu)
    
    if K2 is None:
        K2 = np.linalg.inv(S2)
        
    diff = mu2 - mu
    
    A1 = K2.dot(diff).dot(diff)
    A2 = np.trace(K2@S)    
    A3 = np.linalg.slogdet(S)[1] - np.linalg.slogdet(S2)[1]
    
    KL = 0.5*(A1 + A2 - A3 - p)
    
    return KL