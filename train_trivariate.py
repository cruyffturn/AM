# -*- coding: utf-8 -*-
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras

#import helper_adv
#import helper
import helper_data
import helper_dag
import helper_tf_model

#%%
N = 1000
seed = 65
print('tf seed', seed)
tf.random.set_seed(seed)

#%%
mode = 1

X, mu, \
S, sigma_sq, \
W, data_dic = helper_data.get_data_dag(N, mode, seed_np = 42)

#%% Sets the adv. parameter
W_a = copy.deepcopy(W)
W_a[0,1] = 0
S_a = helper_dag.get_cov(W_a, sigma_sq)

mu_a = mu
idx_adv_train = np.array([0,1])
#%%
   
p = X.shape[1]
model_cfg = 4
loss_type = 9
bool_full = 1
bool_draw = 0
bool_ratio = 0

n_steps = 40
reg_lmbda = 1e-2 #test

model = helper_tf_model.get_model(p, model_cfg, 
                                  loss_type,
                                  idx_adv_train,
                                  bool_full,
                                  mu = mu, S = S,
                                  mu_a = mu_a, S_a = S_a,
                                  bool_ratio = bool_ratio)

model._set_param(S, mu,S_a, mu_a, n_steps, loss_type, bool_full,
                 bool_draw = bool_draw,
                 idx_adv = idx_adv_train,
                 reg_lmbda = reg_lmbda,
                 bool_sub = True)
    
lr = .01
model.compile(
              optimizer=keras.optimizers.Adam(lr),
              run_eagerly = 1,
              )

min_delta = 1e-4
    
history = model.fit(X, 
                    epochs=300,
                    batch_size = N,
                    callbacks = [keras.callbacks.EarlyStopping(monitor='loss', 
                                                               min_delta = min_delta,
                                                               patience=10,
                                                               verbose=1,
                                                               restore_best_weights=True)],   
                    )


#%%
import os
import inspect

import helper_draw
filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))


#%%

fig = helper_draw.draw_loss(history, loss_type, False)

dict_sub = dict(
                model_cfg=model_cfg,
                loss_type=loss_type,
                bool_full=bool_full,
                mode=mode,
                seed=seed,
                dag=True)
dict_sub = {**dict_sub,**data_dic}

figname = '_'.join(['%s_%s'%(a,b) for a, b in dict_sub.items()])

savePath = os.path.join(currPath, 'train_'+figname)
if not os.path.exists( savePath):
	os.makedirs( savePath)

fig.savefig(os.path.join(savePath,'tri_'+figname+'.png'), 
	            dpi=200, bbox_inches='tight')

model.save(os.path.join(savePath,'model'))
