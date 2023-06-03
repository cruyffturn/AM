# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def gather_x2(A, idx_r, idx_c):
    
    B1 = tf.gather(A, indices=idx_r)
    B = tf.gather(B1, indices=idx_c, axis=1)
    
    return B

def get_cond_prob(X_o, idx_1, idx_2, mu, S):
                 
    '''
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
    
    mu_2 = tf.gather(mu, indices=idx_2).numpy()
    mu_1 = tf.gather(mu, indices=idx_1).numpy()
    
    #Calculating the sub cov. matrices
    
    S_11 = gather_x2(S, idx_1, idx_1)
    S_12 = gather_x2(S, idx_1, idx_2)
    S_21 = tf.transpose(S_12)
    S_22 = gather_x2(S, idx_2, idx_2)
    
#    import ipdb;ipdb.set_trace()
#    if S_22.shape[0] > 2:
    S_22_inv = tf.linalg.inv( S_22)                  #It's used twice.
    
#    elif S_22.shape[0] == 1:
#        print('s22',S_22.numpy()[0,0])
#        S_22_inv = tf.ones(S_22.shape[0])/S_22
#    else:
#        temp = tf.reverse(tf.transpose(S_22),1) * (tf.eye(1)-tf.reverse(tf.eye,1))
#        det = tf.math.reduce_prod(tf.linalg.diag(S_22)) \
#            - tf.math.reduce_prod(tf.linalg.diag(tf.reverse(S_22,1)))
#        S_22_inv = temp/det
        
#    S_22_inv = tf.linalg.inv( S_22+1e-3*tf.eye(S_22.shape[0]))
    
    
    A1 = S_12 @ S_22_inv                                #D_1xD_1 * p2xp2
    A2 = x_2 - mu_2
    A3 = A1 @ tf.transpose(A2)
        
    mu_1_2 = tf.transpose(A3) + mu_1
    
    S_1_2 = S_11 - A1 @ S_21                        #D_1xD_1    
    
    return mu_1_2, S_1_2

def get_KL(mu, S, mu2, S2, K2=None):
    
    '''
    https://statproofbook.github.io/P/mvn-kl.html
    Eq. (11)
    '''
    
    if K2 is None:
        K2 = tf.linalg.inv(S2)
    
    p = mu.shape[0]
    
    diff = mu2 - mu
    
    A1 = tf.tensordot(tf.tensordot(K2,diff,1),diff,1)
    A2 = tf.linalg.trace(K2@S)
    A3 = tf.linalg.slogdet(S)[1] - tf.linalg.slogdet(S2)[1]
    
    KL = 0.5*(A1 + A2 - A3 - p)
    
    return KL

def get_entropy(S):
    
    p = S.shape[0]
    
    c = p*(tf.math.log(2*np.pi)+1)
    
    entropy = 0.5*(c + tf.linalg.slogdet(S)[1])

    return entropy

def get_like(X, mu, S):
    
    '''
    In:
        X:  N,p
        mu: p,
        S:  p,p
        
    Inter:
        diff:   N,p
    '''
    
    p = X.shape[1]
    
    A1 = -p/2*np.log(2*np.pi)
    
    A2 = -0.5*tf.linalg.slogdet(S)[1]
    
    K = tf.linalg.inv(S)
    
    diff = X - mu
    A3_1 = diff @ K     #N,p
    A3 = -0.5*tf.math.reduce_sum(A3_1 * diff,1)
    
    log_like = A1 + A2 + A3
    
    like_p = scipy.stats.multivariate_normal.pdf(X.numpy(), mu, S)
    
    return log_like