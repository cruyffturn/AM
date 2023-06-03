# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

import helper_bnlearn
import helper_dag
import helper_tf_model

import os
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))

#%%
#name = 'magic-irri'
name = 'magic-niab'

if name == 'magic-niab':
    seed = 630
elif name == 'magic-irri':
    seed = 397

#seed = 100

print('tf seed', seed)
tf.random.set_seed(seed)

#%%
mode = 1            #don't care
mode_adv = None     #don't care
bool_l1 = 0
bool_hat_true = None    #don't care

data_dic = {'set':'bnlearn_'+name}

X, mu, W, sigma_sq, columns = helper_bnlearn.get_data(name)

S = helper_dag.get_cov(W, sigma_sq)

#Gets the adv. parameters
W_a, sigma_sq_a, \
idx_adv_train, attack_node = helper_bnlearn.get_W_a(X, W, sigma_sq, 
                                                    name, columns)

S_a = helper_dag.get_cov(W_a, sigma_sq_a)
mu_a = mu

N = len(X)

P_a = helper_dag.get_pear(S_a)
P = helper_dag.get_pear(S)

#%%

p = X.shape[1]
model_cfg = 4
loss_type = 9
bool_full = 1
bool_draw = 0
bool_ratio = 0
reg_lmbda = 1e-2
n_steps = 40
bool_sub = True

bool_retrain = False
bool_force_zero = 0

savePath = helper_tf_model.get_savePath(model_cfg, loss_type, 
                                        bool_full, mode,data_dic,
                                        reg_lmbda, seed,
                                        currPath,
                                        bool_retrain,
                                        mode_adv,
                                        bool_l1,
                                        attack_node,
                                        bool_hat_true,
                                        bool_sub,
                                        bool_force_zero)

if not bool_retrain:
    model = helper_tf_model.get_model(p, model_cfg, 
                                      loss_type,
                                      idx_adv_train,
                                      bool_full,
                                      mu = mu, S = S,
                                      mu_a = mu_a, S_a = S_a,
                                      bool_ratio = bool_ratio,
                                      X = X,
                                      bool_sub = bool_sub)
    
    model._set_param(S, mu,S_a, mu_a, n_steps, loss_type, bool_full,
                     bool_draw = bool_draw,
                     idx_adv = idx_adv_train,
                     reg_lmbda = reg_lmbda,
                     bool_l1 = bool_l1,
                     bool_sub = bool_sub)        
    
    if name == 'magic-irri':
        lr = 0.003       
    else:
        lr = 0.001
    
    model.compile(
                  optimizer=keras.optimizers.Adam(lr),
                  run_eagerly = 1,
                  )
        
min_delta = 1e-5

try:
    os.makedirs(os.path.join(savePath,'checkpoint'))
except:
    pass
callbacks = [tf.keras.callbacks.ModelCheckpoint(
    os.path.join(savePath,'checkpoint'),
    monitor='avg_loss',
    verbose=1,
    save_best_only=True,
)]

             
if name == 'magic-irri':
    epochs = 500
else:
    epochs=300
    
#epochs = 1
history = model.fit(X, 
#                    epochs=1,
                    epochs=epochs,
                    batch_size = N,
                    callbacks=callbacks
                    )
    
#%%
import os
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))


#import matplotlib.pyplot as plt
import helper_draw

fig = helper_draw.draw_loss(history, loss_type, False)

savePath = helper_tf_model.get_savePath(model_cfg, loss_type, 
                                        bool_full, mode,data_dic,
                                        reg_lmbda, seed,
                                        currPath,
                                        bool_retrain,
                                        mode_adv,
                                        bool_l1,
                                        attack_node,
                                        bool_hat_true,
                                        bool_sub,
                                        bool_force_zero)


if not os.path.exists( savePath):
	os.makedirs( savePath)
    

if 1:
    fig.savefig(os.path.join(savePath,'train_loss'+'.png'), 
    	            dpi=200, bbox_inches='tight')
        #
    import pickle
    with open(os.path.join(savePath,'history.p'), "wb") as f:
        pickle.dump([history.history], f)
    with open(os.path.join(savePath,'optim.p'), "wb") as f:
        pickle.dump([model.optimizer.get_weights(),
                     model.optimizer.get_config()], f)
        
    model.save(os.path.join(savePath,'model'))
