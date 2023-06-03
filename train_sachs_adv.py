# -*- coding: utf-8 -*-
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras

import os
import inspect

import helper_data
import helper_tf_model
import helper_draw


filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))


#%%
seed = 812
print('tf seed', seed)
tf.random.set_seed(seed)

#%%
mode = 1

mode_adv = 0
bool_l1 = 0
bool_hat_true = True

data_dic={'set':'sachs'}

X, mu, S = helper_data.get_sachs()
N = len(X)

attack_node = 'pip2'
bool_force_zero = False

if mode_adv == 0:
    
    S_a = copy.deepcopy(S)
    
    columns = ['raf', 'mek', 'plc', 
               'pip2', 'pip3', 'erk', 
               'akt', 'pka', 'pkc', 
               'p38', 'jnk']
    
    
    for name_1, name_2 in [['plc','pip2'],
                           ['plc','pip3']]:
        idx_r = columns.index(name_1)
        idx_c = columns.index(name_2)
        
        S_a[idx_r,idx_c] = 0
        S_a[idx_c,idx_r] = 0    
    
    mu_a = mu
    idx_L = [columns.index(name_i) for name_i in ['plc','pip2','pip3']]

    idx_adv_train = np.array(idx_L)

#%%

p = X.shape[1]
model_cfg = 4
loss_type = 9
bool_full = 1
bool_draw = 0
bool_ratio = 0
reg_lmbda = 1e-25
n_steps = 40
bool_sub = True

bool_retrain = False

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
#                                      X = X,
                                      bool_sub = bool_sub)
    
    model._set_param(S, mu,S_a, mu_a, n_steps, loss_type, bool_full,
                     bool_draw = bool_draw,
                     idx_adv = idx_adv_train,
                     reg_lmbda = reg_lmbda,
                     bool_l1 = bool_l1,
                     bool_sub = bool_sub)
        
    lr = 0.001
    model.compile(
                  optimizer=keras.optimizers.Adam(lr),
                  run_eagerly = 1,
                  )
        

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

history = model.fit(X, 
                    epochs=300,
                    batch_size = N,
                    callbacks=callbacks
                    )
    
#%%

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
    

fig.savefig(os.path.join(savePath,'train_loss'+'.png'), 
	            dpi=200, bbox_inches='tight')

import pickle

with open(os.path.join(savePath,'history.p'), "wb") as f:
    pickle.dump([history.history], f)
with open(os.path.join(savePath,'optim.p'), "wb") as f:
    pickle.dump([model.optimizer.get_weights(),
                 model.optimizer.get_config()], f)
    
model.save(os.path.join(savePath,'model'))