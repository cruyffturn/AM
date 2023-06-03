# -*- coding: utf-8 -*-
import numpy as np

from helper_mvn import get_ratio_ideal

def get_prob_rs(X,
                mu_a, S_a,
                mu, S,
                idx_adv,
                seed = 42):
    

    S_sub = S[np.ix_(idx_adv,idx_adv)]
    mu_sub = mu[idx_adv]
    
    S_a_sub = S_a[np.ix_(idx_adv,idx_adv)]
    mu_a_sub = mu_a[idx_adv]
    
    X_sub = X[:,idx_adv]


    ratio = get_ratio_ideal(mu_a_sub, S_a_sub, 
                            mu_sub, S_sub, 
                            X_sub)
        

    pr = ratio/np.max(ratio)        
        
    p2 = np.zeros((len(X),int(2**len(idx_adv))))
    
    p2[:,0] = pr
    p2[:,-1] = 1-pr    
    
    return p2