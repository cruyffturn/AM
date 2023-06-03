# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression

def get_cov(W, sigma_sq):
    
    p = len(W)
    
    A = np.linalg.inv(np.eye(p)-W.T)
    
    if np.isscalar(sigma_sq):
        S = sigma_sq*A @ A.T
    else:        
        S = A @ np.diag(sigma_sq) @ A.T
    
    return S

def get_W_from_dag(X, A, bool_ev = False):
    
    '''
    Inspried by peters 2014 
    "Identifiability of Gaussian structural equation models with equal error variances"
    
    In:
        X:  N,p
        A:  p,p        
    '''
    
    p = len(A)
    
    W = np.zeros([p,p])
    if not bool_ev:
        sigma_sq = np.zeros([p])
    else:
        sigma_sq = 0
    
    for j in range(X.shape[1]):
        
        idx_pa = np.where(A[:,j])[0]
        if len(idx_pa) != 0:
            model = LinearRegression()
            model.fit(X[:,idx_pa],X[:,j])
            
            W[idx_pa,j] = model.coef_
            
            res = np.mean((model.predict(X[:,idx_pa])-X[:,j])**2)            
                
        else:
#            sigma_sq[j] = np.std(X[:,j])**2
            res = np.std(X[:,j])**2
            
        if not bool_ev:
            sigma_sq[j] = res
        else:
            sigma_sq += res
        
    return W, sigma_sq

def get_pear(S):
    
    pear = S/np.sqrt(np.outer(np.diag(S),np.diag(S)))
    
    return pear

def sample_dag(W, sigma_sq, N = 1000):
    
    '''
    Inspried by peters 2014 
    "Identifiability of Gaussian structural equation models with equal error variances"
    
    In:
        X:  N,p
        A:  p,p        
    '''
    
    p = len(W)
    A = np.abs(W)>0

    Z = np.random.normal(size=[N,p])
    X = np.full([N,p], np.nan)
#    X = np.nan
    done = np.zeros(p, bool)

    while not np.all(done):
        
#        import ipdb;ipdb.set_trace()
        for j in range(p):
            
            if not done[j]:
                idx_pa = np.where(A[:,j])[0]
                
                if len(idx_pa) == 0 or np.all(done[idx_pa]):
                    
                    if np.isscalar(sigma_sq):
                        noise = Z[:,j]*np.sqrt(sigma_sq)
                    else:
                        noise = Z[:,j]*np.sqrt(sigma_sq[j])
                        
                    if len(idx_pa) != 0:
                        X[:,j] = X[:,idx_pa].dot(W[idx_pa,j]) + noise
                    else:
                        X[:,j] = noise
                    done[j] = True
        
    return X