# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
import copy

from joblib import Parallel, delayed, parallel_backend
#%%

import helper_data
import helper_dag
import helper_em_tf
import helper_rs
import helper

#%% Parameters
bool_mcar = False   #True: MNAR, False:MCAR
exp_type = 0        #0: MissDAG, 1:missPC, imputation

#%% Generating the data
np.random.seed(2)

edge_L = [[0,1],
          [0,3],
          [1,2],
          [2,3],
          [2,4],
          [2,5]]
p = 6
W = np.zeros([p,p])

for idx_r, idx_c in edge_L:
    
    if np.random.binomial(1,0.5):
        W[idx_r,idx_c] = np.random.uniform(0.5,2)
    else:
        W[idx_r,idx_c] = -np.random.uniform(0.5,2)
    
W[1,2] = 0.4


sigma_sq = 1
N = 50000
X, S = helper_data.sample_dag(W, N, 
               sigma_sq,
               seed = 24, 
               return_S = True)

mu = np.zeros(len(S))

#%% Selecting the adv. parameter

idx_adv_train = np.array([1,2])

W_a = copy.deepcopy(W)
W_a[1,2] = 0.25

S_a = helper_dag.get_cov(W_a, sigma_sq)
mu_a = mu

#%% Local Rejection Sampling attack

p_r_x = helper_rs.get_prob_rs(X,
                      mu_a, S_a,
                      mu, S,
                      idx_adv_train,
                      seed = 42)

#%% Samples the masks
bool_full = 1

if not bool_mcar:
    if not bool_full:
        p_r_x = p_r_x.reshape(-1)
        p2 = np.stack([1-p_r_x,p_r_x],1)
    else:
        p2 = p_r_x
else:
    if bool_full:
        p_r = p_r_x.mean(0)[np.newaxis,:]
        p2 = np.repeat(p_r,N,0)
    

#%% N_rep
      
n_rep = 20


df_L = []
import pandas as pd

label0 = 'lb'
label1 = r'KL($\hat{\theta}||\theta_p$)'
label2 = r'KL($\hat{\theta}||\theta_{\alpha}$)'
label3 = r'HD($\hat{\mathcal{G}},\mathcal{G}_{\alpha}$)'
label4 = r'HD($\hat{\mathcal{G}},\mathcal{G}_{p}$)'
label5 = 'Adv. Success'
hue = r'Init./$\epsilon$'

bool_init = 1
if bool_init:
    init_L = [0,1,3,4,6]    
    eps_L = [.001,]


            
#%%
A_p = (np.abs(W)>0).astype(int)
A_a = copy.deepcopy(A_p)
A_a[1,2] = 0
if exp_type == 0:
    init_dic = {0:'True',
                1:'Ident.',
                3:'Random(*)',
                4:'Emp. Diag.',
                5:'Random',
                6:'IW(*)'
                }
    
    bool_history = 0
    like_all_l_param = []
    bool_sparse  = 0
    bool_dag = 1    

    temp_folder = None
    n_jobs = 1#int(os.environ['MAX_CORES'])
    with parallel_backend('multiprocessing',
    	                      n_jobs = n_jobs):
                            
        
        trainable = delayed(helper.multiple_em)
        load_L = Parallel(verbose=11,
        	                      temp_folder=temp_folder)(trainable(X, p2, n_rep,
                                                          init_mode,
                                                          bool_full, S, mu,
                                                           idx_adv = idx_adv_train,
                                                           bool_sparse = bool_sparse,
                                                           alpha = None,
                                                           bool_while = True,
                                                           eps = eps,
                                                           bool_history = bool_history,
                                                           bool_dag = bool_dag,
                                                           bool_ev = True)
        	                    for init_mode in init_L
                                    for eps in eps_L)
                        
    hue_l = [(init_mode,eps) for init_mode in init_L for eps in eps_L]
    for load,(init_mode,eps) in zip(load_L,hue_l):
        if not bool_history:
            
            if not bool_dag:
                pass                
            else:
                mu_est_all, S_est_all, \
                W_est_all, lb_all = load
                

                A_est_all = (np.abs(W_est_all)>0).astype(int)
                
                n_total_a = helper.get_pdag_dist(A_a, A_est_all)
                n_total_p = helper.get_pdag_dist(A_p, A_est_all)
                
                err_rate = helper.get_adv_err(A_p, A_a, A_est_all)
                succ_rate = 1-err_rate
                
            stats = helper.get_stats(mu_est_all, S_est_all,
                                     mu_a, S_a, 
                                     mu, S)            
            
            df = pd.DataFrame(stats, columns = [label1, label2])
        
            if eps == .001:
                eps_str = 'same'
            elif eps < .001:
                eps_str = 'strict'
            elif eps > .001:
                eps_str = 'loose'
                
            df[hue] = init_dic[init_mode]+'/'+ eps_str# + '/'+f'{eps/100:.0E}'
            df[label3] = n_total_a
            df[label4] = n_total_p
            df[label5] = succ_rate
            df[label0] = lb_all
            
            df_L.append(df)
        else:
            like_all_l = load
            like_all_l_param.append(like_all_l)

else:   #Exp type 1
    hue_est = 'impute/method'
    df_l_2 = []
    
    for bool_impute in [0,1]:
        
        if not bool_impute:
            mode_est_L = ['pc']
        else:
            mode_est_L = ['pc','nt']
        
        for mode_est in mode_est_L:
            A_est_all = helper.multiple_est(X, p2, n_rep,
                                            bool_full, 
                                            bool_impute,
                                            mode_est,
                                            idx_adv = idx_adv_train,
                                            lambda1 = 0
                                            )
            
            if mode_est == 'nt':
                A_est_all = (np.abs(A_est_all)>0).astype(int)            
                    
            n_total_a = helper.get_pdag_dist(A_a, A_est_all, allow_pdag = True)
            n_total_p = helper.get_pdag_dist(A_p, A_est_all, allow_pdag = True)                    
                        
            err_rate = helper.get_adv_err(A_p, A_a, A_est_all, 
                                          allow_pdag = True)
            succ_rate = 1-err_rate
            
            df = pd.DataFrame(np.stack([n_total_a,n_total_p,succ_rate],1), 
                              columns = [label3, label4,label5])
            
            df[hue_est] = str(bool(bool_impute))+'/'+mode_est
            
            df_l_2.append(df)
#%%       
import os
import inspect
import matplotlib.pyplot as plt
import seaborn as sns

filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))

savePath = os.path.join(currPath,'rs')
savePath2 = os.path.join(savePath,'fig')

if bool_mcar:
    savePath2 = os.path.join(savePath2,'mcar')
    
if not os.path.exists( savePath2):
	os.makedirs( savePath2)
    

if bool_full:
    p_r = p_r_x.mean(0)
    obs_mat = helper_em_tf._get_obs_mat(len(idx_adv_train)).astype(int)
    row_L = [','.join(row.astype(str).tolist()) for row in obs_mat]
df_prob = pd.DataFrame({'observed indices (%s)'%idx_adv_train:row_L,
                        'probability':p_r})
df_prob.to_csv(os.path.join(savePath2,'p_r.csv'),
                       index=False)

    
if exp_type == 0:
    
    with open(os.path.join(savePath2,'results_exp_type_0.p'), "wb") as f:
    	pickle.dump(df_L, f)

    figname = ''
    if bool_sparse:
        figname = 'sparse_' + figname
        
    if bool_init:
        figname = 'init_'
    else:
        figname = 'eps_'
    if not bool_history:	
        df_all = pd.concat(df_L, ignore_index = 1)

        df_mean = df_all.groupby(hue).mean()
        df_std = df_all.groupby(hue).std()
        
        summary = pd.concat([df_mean,df_std], axis = 1)
        summary.to_csv(os.path.join(savePath2,'summary.csv'))
        
        n_ax = 4
        fig, ax = plt.subplots(1,n_ax)
        
        sns.scatterplot(data=df_all, x = label1, y = label2, hue = hue,
                        ax = ax[0])
        
        sns.countplot(data=df_all, x = hue, hue = label3,
                      ax = ax[1])
        
        sns.countplot(data=df_all, x = hue, hue = label4,
                      ax = ax[2])
                
        sns.countplot(data=df_all, x = hue, hue = label5,
                          ax = ax[3])            
        
        fig.set_size_inches( w = n_ax*10,h = 5)	
        fig.savefig(os.path.join(savePath2,figname+'denea.png'), 
        	            dpi=200, bbox_inches='tight')
        
    else:
        fig, ax = plt.subplots()
        i = 0
        import matplotlib.cm as cm
        total = len(like_all_l_param)*n_rep
        for like_all_l in like_all_l_param:
            for like_ in like_all_l:
                c = cm.Blues(i/total,1)
                ax.plot(like_, color=c)
                i+=1
                
        fig.savefig(os.path.join(savePath2,figname+'like_denea.png'), 
        	            dpi=200, bbox_inches='tight')
    
else:   #Exp type 1
    
    with open(os.path.join(savePath2,'results_exp_type_1.p'), "wb") as f:
    	pickle.dump(df_l_2, f)
        
    df_all = pd.concat(df_l_2, ignore_index = 1)
    
    df_mean = df_all.groupby(hue_est).mean()
    df_std = df_all.groupby(hue_est).std()
        
    summary = pd.concat([df_mean,df_std], axis = 1)
    summary.to_csv(os.path.join(savePath2,'summary_exp_type_1.csv'))
        
    fig, ax = plt.subplots(1,3)
    sns.countplot(data=df_all, x = hue_est, hue = label4,
                  ax = ax[1])
    sns.countplot(data=df_all, x = hue_est, hue = label3,
                  ax = ax[0])
    sns.countplot(data=df_all, x = hue_est, hue = label5,
                  ax = ax[2])
    
    fig.set_size_inches( w = 15,h = 5)
    fig.savefig(os.path.join(savePath2,'2_''denea.png'),
                dpi=200, bbox_inches='tight')
    
    temp = df_all.groupby(hue_est)[label4].value_counts(normalize=1)
    temp.plot(kind='bar',stacked=True)