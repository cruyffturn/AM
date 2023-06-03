# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats

import helper_dag
import helper_mvn

from External.notears.linear import notears_linear


def get_outer_sum(X, Y = None):
    
    '''
    In:
        X:  N,p1
        Y:  N,p2
    Out:
        Z:  p1,p2
    '''
    
    if Y is None:
        Y = X
        
    Z = np.einsum('ij,ik->jk', X, Y)
    
    return Z
    
def sub(X_o, idx_m, idx_o, mu, S):
        
    mu_1_2, S_1_2 = helper_mvn.get_cond_prob(X_o, idx_m, idx_o, 
                                             mu, S)
    
    
def update_total(X_o, idx_m, idx_o, 
                 mu, S, 
                 total_inr,
                 total_x):
    
    '''
    If idx_m is empty, it still works
    In:
        X_o:        N,|o|
        total_inr:  p,p
        total_x:    p,
        
    Inter:
        mu
    '''
    
    mu_1_2, S_1_2 = helper_mvn.get_cond_prob(X_o, idx_m, idx_o, 
                                             mu, S)
    
#    import ipdb;ipdb.set_trace()
    out_o = get_outer_sum(X_o)              #|o|,|o|
    out_cross = get_outer_sum(X_o, mu_1_2)  #|o|,|m|
    
    out_mu = get_outer_sum(mu_1_2)          #|m|,|m|
                
    total_inr[np.ix_(idx_o,idx_o)] += out_o
    total_inr[np.ix_(idx_o,idx_m)] += out_cross    
    
    total_inr[np.ix_(idx_m,idx_o)] += out_cross.T
    total_inr[np.ix_(idx_m,idx_m)] += len(X_o)*S_1_2 + out_mu
    
    total_x[idx_o] += np.sum(X_o, 0)
    total_x[idx_m] += np.sum(mu_1_2, 0)
  
def get_like(X, mu, S, unq, inverse):

    '''
    In:
        X:      N,p
        mu:     p,
        S:      p,p
    '''
    #Loops different missing patterns
    
    log_like = np.zeros(len(X))
    
    for i, unq_ in enumerate(unq):
        
        idx = np.where(inverse == i)[0]        
        idx_o = np.where(~unq_)[0]
        
        if len(idx_o) != 0:
            X_o = X[idx][:,idx_o]
            mu_o = mu[idx_o]
            S_o = S[np.ix_(idx_o,idx_o)]
            
            log_like[idx] = scipy.stats.multivariate_normal.logpdf(X_o, 
                                                          mean=mu_o, 
                                                          cov=S_o,
#                                                          allow_singular=1
                                                          )
        else:
            log_like[idx] = np.nan
            
    return log_like    
        
def get_em(X, mu_0 = None, S_0 = None, 
           bool_sparse = False,
           alpha = 0.1,
           bool_tf = False,
           n_steps = 10,
           verbose = True,
           bool_history = False,
           bool_while = False,
           eps = .001,
           bool_dag = False,
           w_threshold = 0.3,
           bool_ev = False,
           bool_only_last = False,
           **kwargs_dag
#           bool_only_last = True
           ):
    
    '''
    In:
        X:      N,p
        mu_0:   p,
        S_0:    p,p
        
    '''
    
    idx_miss = np.isnan(X)
    
    unq, inverse = np.unique(idx_miss, axis=0, 
                             return_inverse=True)
    
    if not bool_dag:
        mu_prev = mu_0
    else:
        mu_prev = np.zeros_like(mu_0)
        
    S_prev = S_0
    log_like_prev = get_lower_bound(X, unq, inverse, mu_prev, S_prev)
    
    if bool_while:
        n_steps = 1000
        
    for i in range(n_steps):
        
        log_like_new = get_lower_bound(X, unq, inverse, mu_prev, S_prev)
        
        if verbose: print('step%i'%(i+1),log_like_new)
                        
        if not bool_dag or bool_only_last:
            pass
        else:
            W, sigma_sq = sub_em_dag(X, unq, inverse, 
                                     mu_prev, S_prev,
                                     bool_ev = bool_ev,
                                     verbose = verbose,
                                     **kwargs_dag)
            
            if sigma_sq.max()>100:
                print(np.round(sigma_sq,2))
            
            mu_new = np.zeros(X.shape[1])
            S_new = helper_dag.get_cov(W, sigma_sq)
            
        diff = np.abs((log_like_new - log_like_prev)/log_like_prev)*100
        
        mu_prev = mu_new
        S_prev = S_new
        log_like_prev = log_like_new
        
        if bool_history:
            if i == 0:
                like_L = [log_like_prev]
                mu_L = [mu_prev]
                S_L = [S_prev]
            else:
                like_L.append(log_like_prev)
                mu_L.append(mu_prev)
                S_L.append(S_prev)
                
        if bool_while and (diff < eps and i!=0):#.001
            print('stop',i,diff)           
            break

            
    if not bool_history:
        if not bool_dag:
            pass
        else:
            if bool_dag:
                W[np.abs(W)<w_threshold] = 0
#                A = np.linalg.inv(np.eye(len(W))-W.T)
#                S_new = sigma_sq*A@A.T
                S_new = helper_dag.get_cov(W, sigma_sq)
                return W, sigma_sq
    else:
        return like_L, mu_L, S_L

def sub_em(X, unq, inverse, mu, S, return_ss = False):
    
    '''
    In:
        X:      N,p
    
    Inter:
        total_inr:  #expected outer product of x (unnorm Eq. 11.110)
        total_x:    #expected x (unnorm Eq. 11.109)
    '''
        
    N,p = X.shape
    
    total_inr = np.zeros((p,p))
    total_x = np.zeros(p)  

    N_used = 0
    #Loops different missing patterns
    for i, unq_ in enumerate(unq):
        
        idx = np.where(inverse == i)[0]
        
        idx_m = np.where(unq_)[0]
        idx_o = np.where(~unq_)[0]
        
        if len(idx_o) != 0:
            X_o = X[idx][:,idx_o]
    
            update_total(X_o, idx_m, idx_o, 
                         mu, S, 
                         total_inr,
                         total_x)
        
            N_used += len(X_o)
        
#    import ipdb;ipdb.set_trace()
#    print('used', N_used, 'N', N)
    if not return_ss:
        mu_new = total_x/N_used
        S_new = total_inr/N_used - np.outer(mu_new,mu_new)
        
        return mu_new, S_new
    else:
        x = total_x/N_used
        outer = total_inr/N_used
        return x, outer

def sub_em_dag(X, unq, inverse, 
               mu, S, 
               bool_nt = 1,
               bool_ev = True,
               verbose = True,
               lambda1 = 0.1):
    
    x, outer = sub_em(X, unq, inverse, 
                      mu, S, 
                      return_ss = True)


    if bool_nt:
        if lambda1 != 0.1:
            if verbose: print('lambda=', lambda1)

        loss_type = 'l2_miss'
        W = notears_linear(outer, lambda1, loss_type, 
                           max_iter=100, 
                           h_tol=1e-8, 
                           rho_max=1e+16, 
    #                       w_threshold=0.3
                           w_threshold=0
                           )    
        
    p = X.shape[1]
    if bool_ev:
        sigma_sq = _loss_miss(W, outer)/p
    else:
        sigma_sq = _loss_miss_nv(W, outer)
    
#    import ipdb;ipdb.set_trace()
    return W, sigma_sq

def _loss_miss(W, outer):
    
    A = outer
    B = A @ W
    C = np.sum(W*B)
    
    loss = np.trace(A)-2*np.sum(W*A) + C
    
    return loss

def _loss_miss_nv(W, outer):
    
    A = outer
    B = A @ W
    C = np.sum(W*B,0)    
    
    sigma_sq = np.diag(A)-2*np.sum(W*A,0) + C
    
    return sigma_sq
    
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def sub_em_enum(X, inverse, mu, S):
    
    '''
    In:
        X:      N,p
    
    Inter:
        total_inr:  #expected outer product of x (unnorm Eq. 11.110)
        total_x:    #expected x (unnorm Eq. 11.109)
    '''
        
    N,p = X.shape
    
    total_inr = np.zeros((p,p))
    total_x = np.zeros(p)  

    N_used = 0
    #Loops different missing patterns
    for idx_temp in powerset(np.arange(p)):
        
        idx_m = np.array(idx_temp)
        idx_o = np.setdiff1d(np.arange(p), idx_m)

        if len(idx_o) != 0:
            X_o = X[:,idx_o]
    
            update_total(X_o, idx_m, idx_o, 
                         mu, S, 
                         total_inr,
                         total_x)
        
            N_used += len(X_o)
        
#    import ipdb;ipdb.set_trace()
#    print('used', N_used, 'N', N)
    mu_new = total_x/N_used
    S_new = total_inr/N_used - np.outer(mu_new,mu_new)
    
    return mu_new, S_new

def sum_sstats(X, unq, inverse, mu, S):
    
    '''
    In:
        X:      N,p
    
    Inter:
        total_inr:  #expected outer product of x (unnorm Eq. 11.110)
        total_x:    #expected x (unnorm Eq. 11.109)
    '''
      
    N,p = X.shape
        
    total_inr = np.zeros((p,p))
    total_x = np.zeros(p)  

#    N_used = 0
    #Loops different missing patterns
    for i, unq_ in enumerate(unq):
        
        idx = np.where(inverse == i)[0]
        
        idx_m = np.where(unq_)[0]
        idx_o = np.where(~unq_)[0]
        
        if len(idx_o) != 0:
            X_o = X[idx][:,idx_o]
    
            update_total(X_o, idx_m, idx_o, 
                         mu, S, 
                         total_inr,
                         total_x)
        
#            N_used += len(X_o)                                        
            
    return total_inr, total_x

def sum_obj(X, unq, inverse, mu, S, K = None):
    
    '''
    Eq. 11.104
    
    '''
    p = X.shape[1]
    
    N_used = np.sum(~np.all(np.isnan(X),1))
    
    C1 = -0.5*np.linalg.slogdet(S)[1]
    C2 = -0.5*p*np.math.log(2*np.pi)
    
    total_inr, total_x = sum_sstats(X, unq, inverse,
                                    mu, S)
    
    
    avg_x = total_x/N_used
    avg_inr = total_inr/N_used
    
    out_mu = np.outer(mu, mu)
    
    out_mu_x = np.outer(mu, avg_x)
        
#    import ipdb;ipdb.set_trace()
    B1 = avg_inr + out_mu - 2*out_mu_x
    
    if K is None:
        K = np.linalg.inv(S)
        
    C3 = -0.5*np.trace(K @ B1)
    
    obj = C1 + C2 + C3
    
    return obj

def sum_ent(X, unq, inverse, mu, S):
    
    '''
    In:
        X:      N,p
    
    Inter:
        total_inr:  #expected outer product of x (unnorm Eq. 11.110)
        total_x:    #expected x (unnorm Eq. 11.109)
    '''
     
#    p = X.shape[1]
    N_used = np.sum(~np.all(np.isnan(X),1))
    
    sum_ent = 0
    
    #Loops different missing patterns
    for i, unq_ in enumerate(unq):
        
        idx = np.where(inverse == i)[0]
        
        idx_m = np.where(unq_)[0]
        idx_o = np.where(~unq_)[0]
        
        
        if len(idx_o) != 0 and len(idx_m) != 0:
            #entropy of fully observed is zero
            
#            X_o = X[idx][:,idx_o]
            X_o = np.zeros((1,len(idx_o)))
            _, S_1_2 = helper_mvn.get_cond_prob(X_o, idx_m, idx_o, 
                                     mu, S)
            
            entropy = helper_mvn.get_entropy(S_1_2)
            
            sum_ent = sum_ent + len(idx)*entropy
    
    avg_ent = sum_ent/N_used
    
    return avg_ent

def get_lower_bound(X, unq, inverse, mu, S): 
    
#    idx_miss = np.isnan(X)
#    unq, inverse = np.unique(idx_miss, axis=0, 
#                             return_inverse=True)
    
    obj = sum_obj(X, unq, inverse, mu, S, K = None)
    
    avg_ent = sum_ent(X, unq, inverse, mu, S)
    
    lbound = obj + avg_ent
    
    return lbound
            