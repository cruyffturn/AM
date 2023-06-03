
import pickle
import numpy as np
import os
import inspect
import tensorflow as tf
import helper_em_tf, helper_tf_model

import helper_bnlearn
import helper_dag

filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))

#bool_mcar = True
bool_mcar = False
exp_type = 1

#%%

mode = 1            #don't care
mode_adv = None     #don't care
bool_l1 = 0
bool_hat_true = None    #don't care

name = 'magic-niab'
#name = 'magic-irri'

if name == 'magic-niab':
    seed = 630
elif name == 'magic-irri':
    seed = 397
    
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

#%%

p = X.shape[1]
model_cfg = 4
loss_type = 9
bool_full = 1
bool_draw = 0
bool_ratio = 0
#reg_lmbda = 5e-3
reg_lmbda = 1e-2
n_steps = 40
bool_sub = True

bool_retrain = False
bool_force_zero = 0


#%% Sets the seed
np.random.seed(42)

print('tf seed', seed)
tf.random.set_seed(seed)


#%% Gets the tf model
model = helper_tf_model.load_model(model_cfg, loss_type, 
                                   bool_full, mode,data_dic,
                                   reg_lmbda, seed,
                                   currPath,
                                   bool_retrain = bool_retrain,
                                   mode_adv = mode_adv,
                                   bool_l1 = bool_l1,
                                   attack_node = attack_node,
                                   bool_hat_true = bool_hat_true,
                                   bool_sub = bool_sub,
                                   bool_force_zero = bool_force_zero)
    
#%%
if not bool_sub:
    p_r_x = model(X, training=True)  # Forward pass
else:    
    p_r_x = model(X[:,idx_adv_train], training=True)  # Forward pass


#p_r_x = model(X).numpy()

if not bool_mcar:
    if not bool_full:
        p_r_x = p_r_x.reshape(-1)
        p2 = np.stack([1-p_r_x,p_r_x],1)
    else:
        p2 = p_r_x
else:
    if bool_full:
        p_r = p_r_x.numpy().mean(0)[np.newaxis,:]
        p2 = np.repeat(p_r,N,0)
    


#%% N_rep
#init_mode = 4
#n_rep = 50
n_rep = 20
#n_rep = 8
#n_rep = 1
import helper

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
else:
    init_L = [0,3]
    eps_L = [.005, .001,.0005]
            
#%%
A_a = (np.abs(W_a)>0).astype(int)
A_p = (np.abs(W)>0).astype(int)
    
if name == 'magic-niab':
    lambda1 = 0
else:
    lambda1 = 0.1
    
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
    err_W_l = []
    
    bool_sparse = 0
    bool_dag = 1

    from joblib import Parallel, delayed, parallel_backend

    temp_folder = None
    n_jobs = 1
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
                                                           verbose = False,
                                                           lambda1=lambda1)
        	                    for init_mode in init_L
                                    for eps in eps_L)
                        
    hue_l = [(init_mode,eps) for init_mode in init_L for eps in eps_L]
    for load,(init_mode,eps) in zip(load_L,hue_l):
            
        if not bool_history:
            
            if not bool_dag:
                mu_est_all, S_est_all, \
                K_est_all, lb_all = load                
                
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
        #    temp = 
            if eps == .001:
                eps_str = 'same'
            elif eps < .001:
                eps_str = 'strict'
            elif eps > .001:
                eps_str = 'loose'
                
            df[hue] = init_dic[init_mode]+'/'+ eps_str# + '/'+f'{eps/100:.0E}'
            if bool_dag:
                df[label3] = n_total_a
                df[label4] = n_total_p
                df[label5] = succ_rate
            df[label0] = lb_all
            
            
            df_L.append(df)
            
            err_W_l.append((A_est_all!=A_a).mean(0))
        else:
            like_all_l = load
            like_all_l_param.append(like_all_l)

else:   #Exp type 1
    n_jobs = 1#int(os.environ['MAX_CORES'])
    
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
                                            n_jobs = n_jobs,
                                            lambda1=lambda1
                                            )
            
            if mode_est == 'nt':
                A_est_all = (np.abs(A_est_all)>0).astype(int)
            
#            import ipdb;ipdb.set_trace()
            
            
            n_total_a = helper.get_pdag_dist(A_a, A_est_all, allow_pdag = True)
            n_total_p = helper.get_pdag_dist(A_p, A_est_all, allow_pdag = True)
                    
            err_rate = helper.get_adv_err(A_p, A_a, A_est_all, 
                                          allow_pdag = True)
            succ_rate = 1-err_rate
            
            df = pd.DataFrame(np.stack([n_total_a,n_total_p,succ_rate],1), 
                              columns = [label3, label4,label5])
                        
            df[hue_est] = str(bool(bool_impute))+'/'+mode_est
            
            df_l_2.append(df)
#%% Draws the figures and exports the tables
import os
import inspect
import matplotlib.pyplot as plt
import seaborn as sns

filename = inspect.getframeinfo(inspect.currentframe()).filename
currPath = os.path.dirname(os.path.abspath(filename))



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

savePath2 = os.path.join(savePath,'fig')

if bool_mcar:
    savePath2 = os.path.join(savePath2,'mcar')
    
if not os.path.exists( savePath2):
	os.makedirs( savePath2)
    

if bool_full:
    p_r = p_r_x.numpy().mean(0)
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
    #    sns.swarmplot(data=df_all, x = hue, y = label3,
    #                  dodge=True,
    #                  ax = ax[1])
        
        if bool_dag:
            sns.countplot(data=df_all, x = hue, hue = label3,
                          ax = ax[1])
            
            sns.countplot(data=df_all, x = hue, hue = label4,
                          ax = ax[2])
            
            sns.countplot(data=df_all, x = hue, hue = label5,
                          ax = ax[3])
        else:
            figname = figname+'_debug'
                
    #    sns.scatterplot(data=df_all, x = label0, y = label3, hue = hue,
    #                    ax = ax[-1])
        
        
        fig.set_size_inches( w = n_ax*10,h = 5)	
        fig.savefig(os.path.join(savePath2,figname+'denea.png'), 
        	            dpi=200, bbox_inches='tight')
        
#            import seaborn as sns
    #    from helper_plot import plots
        
        fig, ax = plt.subplots(1,len(err_W_l))
        if len(err_W_l) == 1:
            ax = [ax]        
        for ax_i, err_W in zip(ax,err_W_l):
            sns.heatmap(err_W, 
                        xticklabels=columns, 
                        yticklabels=columns,
                        ax = ax_i,
                        square = True,
                        center= 0.5,
                        vmin = 0,
                        vmax = 1.,
                        annot = True,
                        fmt=".2f",
            #            fmt = fmt,
            #            **kwargs
                        )
        
#        sns.heatmap((A_est_all!=A_a).mean(0),
#                        xticklabels=columns, 
#                        yticklabels=columns,
#                        ax = ax,
#                        square = True,
##                        center= 0.5,
#            #            vmin = vmin,
#            #            vmax = vmax,
#                        annot = True,
#                        fmt=".2f",
#            #            fmt = fmt,
#            #            **kwargs
#                        )
        
        
#        ax.axes.set_aspect('equal')
        #fig.set_size_inches( w = 10,h = 5)
        fig.set_size_inches( w = len(ax)*8,h = 5)
        fig.savefig(os.path.join(savePath2,figname+'W_err.png'), 
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
    #    like_all_l
else:   #Exp type 1
    
    with open(os.path.join(savePath2,'results_exp_type_1.p'), "wb") as f:
    	pickle.dump(df_l_2, f)
        
    df_all = pd.concat(df_l_2, ignore_index = 1)
    
    df_mean = df_all.groupby(hue_est).mean()
    df_std = df_all.groupby(hue_est).std()
        
    summary = pd.concat([df_mean,df_std], axis = 1)
    summary.to_csv(os.path.join(savePath2,'summary_exp_type_1.csv'))


#    df_all = pd.concat(df_l_2, ignore_index = 1)
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