# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from itertools import chain, combinations

from helper_mvn_tf import (get_cond_prob, 
                           get_entropy)

def gather_x2(A, idx_r, idx_c):
    
    B1 = tf.gather(A, indices=idx_r)
    B = tf.gather(B1, indices=idx_c, axis=1)
    
    return B

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def _get_prob(p_r_x, idx_o):

    mask_0 = np.zeros(p_r_x.shape[1],np.float32)
    mask_0[idx_o] = 1
    mask = tf.constant(mask_0)
    
    p_0 = p_r_x*mask + (1-p_r_x)*(1-mask)
    p = tf.math.reduce_prod(p_0, 1)
    
    return p

def get_outer_sum(X, Y = None, weights = None):
    
    '''
    In:
        X:          N,p1
        Y:          N,p2
        weights:    N,
    Out:
        Z:          p1,p2
    '''
    
    if Y is None:
        Y = X
    
    if weights is not None:
        w_sqrt = tf.math.sqrt(weights)
        X = tf.transpose(tf.transpose(X)*w_sqrt)
        Y = tf.transpose(tf.transpose(Y)*w_sqrt)
        
    Z = tf.einsum('ij,ik->jk', X, Y)
    
    return Z

def _update_total(X_o, idx_m, idx_o, 
                 mu, S, 
                 weights = None):
    
    '''
    In:
        X_o:        N,|o|
        total_inr:  p,p
        total_x:    p,
        weights:    N,
    Inter:
        mu
    '''
    
    if len(idx_m) != 0:
        mu_1_2, S_1_2 = get_cond_prob(X_o, idx_m, idx_o, 
                                      mu, S)
            
        out_o = get_outer_sum(X_o, weights = weights)              #|o|,|o| +
        out_cross = get_outer_sum(X_o, mu_1_2, weights = weights)  #|o|,|m| +
        
        out_mu = get_outer_sum(mu_1_2, weights = weights)          #|m|,|m| +
        
        A = tf.concat([out_o,out_cross],1)
        
        B1 = tf.transpose(out_cross)
        
        if weights is None:
            B2 = len(X_o)*S_1_2 + out_mu
        else:        
            B2 = tf.math.reduce_sum(weights)*S_1_2 + out_mu #+
            
        B = tf.concat([B1,B2],1)
        
    #    import ipdb;ipdb.set_trace()
        inr_unsort = tf.concat([A,B],0)
        
        if weights is None:
            C1 = tf.math.reduce_sum(X_o, 0)
            C2 = tf.math.reduce_sum(mu_1_2, 0)
        else:
            C1 = tf.math.reduce_sum(tf.transpose(X_o)*weights, 1)   #+
            C2 = tf.math.reduce_sum(tf.transpose(mu_1_2)*weights, 1)    #+
            
#        x_unsort = tf.concat([tf.math.reduce_sum(X_o, 0),
#                              tf.math.reduce_sum(mu_1_2, 0)],0)
        x_unsort = tf.concat([C1,
                              C2],0)
    
        idx = np.argsort(np.hstack([idx_o,idx_m]))
        
        inr = gather_x2(inr_unsort, idx, idx)    
        x = tf.gather(x_unsort, idx)
    else:
        out_o = get_outer_sum(X_o, weights = weights)              #|o|,|o| +
        inr_unsort = out_o 
        
        if weights is None:
            x_unsort = tf.math.reduce_sum(X_o, 0)
        else:                    
            x_unsort = tf.math.reduce_sum(tf.transpose(X_o)*weights, 1)     #+
                              
        idx = np.argsort(np.hstack([idx_o,idx_m]))
        
        inr = gather_x2(inr_unsort, idx, idx)
        x = tf.gather(x_unsort, idx)
        
    return inr, x


def get_deno(N, p, idx_adv, p_r_x):
    
    deno = tf.zeros(N)
    p_sub = len(idx_adv)
    
    for idx_temp in powerset(np.arange(p_sub)):
        idx_sub_m = np.array(idx_temp).astype(np.int32)
        idx_sub_o = np.setdiff1d(np.arange(p_sub), idx_sub_m).astype(np.int32)
        
        
        idx_m = idx_adv[idx_sub_m]
        idx_o = np.setdiff1d(np.arange(p), idx_m).astype(np.int32)
        
        if len(idx_o) != 0:     #All missing not included
            deno += _get_prob(p_r_x, idx_sub_o)
    
    return deno
def get_weight(p_r_x, idx_o, deno = None):

    '''
    +
    '''
#    import ipdb;ipdb.set_trace()    
    mask_0 = np.zeros(p_r_x.shape[1],np.float32)
    mask_0[idx_o] = 1
    mask = tf.constant(mask_0)
    
    p_0 = p_r_x*mask + (1-p_r_x)*(1-mask)
    
    if deno is None:
        deno = 1-tf.math.reduce_prod(1-p_r_x, 1)
    p = tf.math.reduce_prod(p_0, 1)
    
    weight = p/deno
    
    return weight

def get_prob_both(p_r_x, bool_full, 
                  idx_o = None, i = None):
            
    '''
    +
    '''
    if not bool_full:
        prob_r = _get_prob(p_r_x, idx_o)
    else:
        prob_r = tf.gather(p_r_x, indices=np.array([i]), 
                           axis = 1)        
        prob_r = tf.squeeze(prob_r)
       
    return prob_r

def sub_em_enum(X, 
                mu, S, 
                p_r_x, 
                idx_adv,
                debug =0, bool_full = False):
    
    '''
    +
    In:
        X:      N,p
    
    Inter:
        total_inr:  #expected outer product of x (unnorm Eq. 11.110)
        total_x:    #expected x (unnorm Eq. 11.109)
    '''
      
#    import ipdb;ipdb.set_trace()
    N,p = X.shape
#    N = X.shape[0:1]
#    p = X.shape[1]
        
    total_inr = tf.constant(tf.zeros((p,p),tf.float32))
    total_x = tf.constant(tf.zeros(p))
    
    S = tf.constant(S,dtype='float32')
    mu = tf.constant(mu,dtype='float32')
    
    if debug:
        p_r_x = tf.Variable(initial_value= tf.ones((N,p))/2, 
                          dtype='float32')    
    
#    idx_n_adv = np.setdiff1d(idx_adv, idx_m)
#            
#    N_used = 0
    p_sub = len(idx_adv)
    if not bool_full:            
        deno = get_deno(N, p, idx_adv, p_r_x)
        
    #Loops different missing patterns
    for i, idx_temp in enumerate(powerset(np.arange(p_sub))):
        
        idx_sub_m = np.array(idx_temp).astype(np.int32)
        idx_sub_o = np.setdiff1d(np.arange(p_sub), idx_sub_m).astype(np.int32)
        
        if not bool_full:
            weights = get_weight(p_r_x, idx_sub_o, deno)
            
        else:
#            import ipdb;ipdb.set_trace()
            num = tf.gather(p_r_x, indices=np.array([i]), axis = 1)
            
            if p_sub == p:
                p_miss = tf.gather(p_r_x, indices=np.array([p_r_x.shape[1]-1]), 
                                   axis = 1)
            else:
                p_miss = 0
            
            weights = tf.squeeze(num/(1-p_miss))
        
        
        idx_m = idx_adv[idx_sub_m]
        idx_o = np.setdiff1d(np.arange(p), idx_m).astype(np.int32)
        
        if len(idx_o) != 0:


            X_o = tf.gather(X, indices=idx_o, axis = 1)
            
            inr, x = _update_total(X_o, idx_m, idx_o, 
                                   mu, S,                           
                                   weights = weights
                                   )
#            print(np.mean(weights.numpy()))
#            import numpy as np
            if debug:
                print(idx_o)
                print(inr.numpy()/N)
                print(x.numpy()/N)
#                import ipdb;ipdb.set_trace()
            total_inr = total_inr + inr
            total_x = total_x + x
#            N_used += len(X_o)
                
    mu_new = total_x/N
#    import ipdb;ipdb.set_trace()
    S_new = total_inr/N - tf.tensordot(mu_new,mu_new,0)#np.outer(mu_new,mu_new)
    
    return mu_new, S_new

def get_em_enum(X, p_r_x, 
                idx_adv,
                mu_0 = None, S_0 = None, 
                bool_sparse = False,
                alpha = 0.1,
                n_steps = 10,
                bool_full = False,
                bool_history = False,
                bool_while = False):
    
    '''
    +
    In:
        X:      N,p
        mu_0:   p,
        S_0:    p,p
        
    '''
        
    mu_prev = mu_0
    S_prev = S_0    
    log_like_prev = get_lower_bound(X, mu_prev, 
                                    S_prev, p_r_x, 
                                    bool_full, idx_adv)
    
    if bool_while:
        n_steps = 1000
        
    for i in range(n_steps):
#        log_like = get_like(X, mu_prev, S_prev, unq)
        
#        print(i)
        log_like_new = get_lower_bound(X, mu_prev, 
                                       S_prev, 
                                       p_r_x, bool_full, idx_adv)
                    
#        print(np.nanmean(log_like))
        
        if not bool_sparse:
#            print(S_prev)            
            mu_new, S_new = sub_em_enum(X, mu_prev, 
                                        S_prev, p_r_x,
                                        idx_adv,
                                        bool_full = bool_full)
            
        else:
            mu_new, S_new, K_new = sub_em_sparse(X, unq, inverse, 
                                                 mu_prev, S_prev,
                                                 alpha)
            
        diff = np.abs((log_like_new.numpy() - log_like_prev.numpy())/log_like_prev.numpy())*100

        
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
            
#        if bool_while: print(diff)
        if bool_while and (diff < .001 and i!=0):
#            print(log_like_new,log_like_prev)
            print('stop',i,diff)
            break
    
    if not bool_history:    
        if not bool_sparse:
            return mu_new, S_new
        else:
            return mu_new, S_new, K_new
    else:
        return like_L, mu_L, S_L
        
def get_mask_full(cat, p, idx_adv):
    
    '''
    +
    '''
#    p = np.log2(p_r_x.shape[1])
    p_sub = len(idx_adv)
    
    set_size = int(2**p_sub)
    bool_miss = np.zeros((set_size,p_sub), bool)
    
    for i, idx_temp in enumerate(powerset(np.arange(p_sub))):
                
        idx_sub_m = np.array(idx_temp).astype(np.int32)        
        bool_miss[i,idx_sub_m] = True
    
#    import ipdb;ipdb.set_trace()
    mask_sub = np.take_along_axis(bool_miss, cat[:,np.newaxis],0)
    
    mask = np.zeros((len(cat),p),bool)
    mask[:,idx_adv] = mask_sub
#    mask = ~mask_miss
    
    return mask

def get_mask_nfull(cat, p, idx_adv):
    
    '''
    +
    '''
    p_sub = len(idx_adv)
    mask_sub = cat.reshape(-1,p_sub).astype(bool)#[:,0]
    mask_sub = ~mask_sub
    
    mask = np.zeros((len(mask_sub),p),bool)
    mask[:,idx_adv] = mask_sub
    
    return mask
            
#%%
def sum_sstats(X, mu, S, p_r_x, bool_full, idx_adv, debug = 0):
    
    '''
    +
    In:
        X:      N,p
    
    Inter:
        total_inr:  #expected outer product of x (unnorm Eq. 11.110)
        total_x:    #expected x (unnorm Eq. 11.109)
    '''
      
    N,p = X.shape
        
    total_inr = tf.constant(tf.zeros((p,p),tf.float32))
    total_x = tf.constant(tf.zeros(p))
    
    if debug:
        p_r_x = tf.Variable(initial_value= tf.ones((N,p))/2, 
                          dtype='float32')                
            
    p_sub = len(idx_adv)
    #Loops different missing patterns
    for i, idx_temp in enumerate(powerset(np.arange(p_sub))):
        
        idx_sub_m = np.array(idx_temp).astype(np.int32)
        idx_sub_o = np.setdiff1d(np.arange(p_sub), idx_sub_m).astype(np.int32)
         
        prob_r = get_prob_both(p_r_x, bool_full, 
                               idx_sub_o, i)
        
        idx_m = idx_adv[idx_sub_m]
        idx_o = np.setdiff1d(np.arange(p), idx_m).astype(np.int32)
        
        if len(idx_o) != 0:

            X_o = tf.gather(X, indices=idx_o, axis = 1)
            
            inr, x = _update_total(X_o, idx_m, idx_o, 
                                   mu, S,                           
                                   weights = prob_r
                                   )

            if debug:
                print(idx_o)
                print(inr.numpy()/N)
                print(x.numpy()/N)
#                import ipdb;ipdb.set_trace()
            total_inr = total_inr + inr
            total_x = total_x + x
            
    return total_inr, total_x

def get_alpha(p, p_r_x, bool_full, idx_adv):
    
    '''
    +
    '''
#    N,p = p_r_x.shape        
    total_alpha = tf.constant(tf.zeros(1,tf.float32))
            
    p_sub = len(idx_adv)
    #Loops different missing patterns
    for i, idx_temp in enumerate(powerset(np.arange(p_sub))):
        
        idx_sub_m = np.array(idx_temp).astype(np.int32)
        idx_sub_o = np.setdiff1d(np.arange(p_sub), idx_sub_m).astype(np.int32)

        idx_m = idx_adv[idx_sub_m]
        idx_o = np.setdiff1d(np.arange(p), idx_m).astype(np.int32)
        
        if len(idx_o) != 0:        

            prob_r = get_prob_both(p_r_x, bool_full, 
                                   idx_sub_o, i)
            
            total_alpha = total_alpha + tf.math.reduce_mean(prob_r)
        
    total_alpha = tf.squeeze(total_alpha)
    
    return total_alpha

def sum_obj(X, mu, S, p_r_x, bool_full, idx_adv, K = None):
    
    '''
    +
    Eq. 11.104
    
    '''
    N,p = X.shape    
    
    total_inr, total_x = sum_sstats(X, mu, 
                                    S, p_r_x, 
                                    bool_full,
                                    idx_adv,
                                    debug = 0)
    
    
    avg_x = total_x/N
    avg_inr = total_inr/N
    
    out_mu = tf.tensordot(mu, mu, 0)
    
    out_mu_x = tf.tensordot(mu, avg_x, 0)
    
    alpha = get_alpha(p, p_r_x, bool_full, idx_adv)
#    import ipdb;ipdb.set_trace()
    B1 = avg_inr + alpha*out_mu - 2*out_mu_x
    
    if K is None:
        K = tf.linalg.inv(S)
        
    C3 = -0.5*tf.linalg.trace(K @ B1)
    
    C1 = -0.5*alpha*tf.linalg.slogdet(S)[1]
    C2 = -0.5*alpha*p*tf.math.log(2*np.pi)
    
    obj = C1 + C2 + C3
    
    return obj

def sum_ent(X, mu, S, p_r_x, bool_full, idx_adv, debug = 0):
    
    '''
    +
    In:
        X:      N,p
    
    Inter:
        total_inr:  #expected outer product of x (unnorm Eq. 11.110)
        total_x:    #expected x (unnorm Eq. 11.109)
    '''
      
    N,p = X.shape
        
    avg_ent = tf.constant(0,tf.float32)
        
    if debug:
        p_r_x = tf.Variable(initial_value= tf.ones((N,p))/2, 
                          dtype='float32')                
    
    p_sub = len(idx_adv)
    #Loops different missing patterns
    for i, idx_temp in enumerate(powerset(np.arange(p_sub))):
        
        idx_sub_m = np.array(idx_temp).astype(np.int32)
        idx_sub_o = np.setdiff1d(np.arange(p_sub), idx_sub_m).astype(np.int32)
        

        prob_r = get_prob_both(p_r_x, bool_full, 
                               idx_sub_o, i)
        
        idx_m = idx_adv[idx_sub_m]
        idx_o = np.setdiff1d(np.arange(p), idx_m).astype(np.int32)

        if len(idx_o) != 0 and len(idx_m) != 0:
            #entropy of fully observed is zero
            
            X_o = tf.gather(X, indices=idx_o, axis = 1)
            
            _, S_1_2 = get_cond_prob(X_o, idx_m, idx_o, 
                                     mu, S)
            
            entropy = get_entropy(S_1_2)
            mean_prob_r = tf.math.reduce_mean(prob_r)
            
            avg_ent = avg_ent + mean_prob_r*entropy
    
    return avg_ent

def get_lower_bound(X, mu, S, p_r_x, bool_full, idx_adv):

    '''
    +
    '''    
    S = tf.constant(S,dtype='float32')
    mu = tf.constant(mu,dtype='float32')
    
    obj = sum_obj(X, mu, S, p_r_x, bool_full, idx_adv, K = None)
    
    avg_ent = sum_ent(X, mu, S, p_r_x, bool_full, idx_adv, debug = 0)
    
    lbound = obj + avg_ent

    return lbound

def get_graph_error(K, K_est_all, thres):
    
    E_est_all = np.abs(K_est_all) > thres
    E = np.abs(K) > thres
    
    n_rem = np.sum(E & (~E_est_all), (1,2))/2   #Division by 2 as symmetric
    n_add = np.sum((~E) & E_est_all, (1,2))/2
    
    return n_rem, n_add


def get_obs_prob(p_r_x, p_sub):
    
    '''
    +
    In:
        p_r_x:      N,2**p_sub
    '''    
    p_r = tf.math.reduce_mean(p_r_x, 0, keepdims=True)
    
    A = _get_obs_mat(p_sub)
#    print(A)
    A = tf.constant(A)
#    import ipdb;ipdb.set_trace()
    return tf.math.reduce_mean(p_r@A)
#    p_r*
#    pass

def _get_obs_mat(p_sub):
    
    '''
    +
    '''
    A = np.zeros([int(2**p_sub),p_sub], np.float32)
    
    for i, idx_temp in enumerate(powerset(np.arange(p_sub))):
        
        idx_sub_m = np.array(idx_temp).astype(np.int32)
        idx_sub_o = np.setdiff1d(np.arange(p_sub), idx_sub_m).astype(np.int32)
        
        A[i,idx_sub_o] = 1
        
    return A
    