# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def draw_loss(history, loss_type, bool_bivar = True):
    
    if bool_bivar:
        n_fig = 3
    else:
        n_fig = 2
        
    fig, ax = plt.subplots(1,n_fig)
    

    #ax.axhline(S[0,1],
    #           xmax=0.5,
    #           label = 'adv. target value'+sub_str,
    #           alpha= 0.2,
    ##           color = clrs[ii]
    #           )
#    if 
    if loss_type == 9 and 'plain_loss' in history.history.keys():
        ax[-2].plot(history.history['plain_loss'])                    
    
    ax[-2].set_xlabel('Training Epoch')
    
    if loss_type == 9:
        if 'plain_loss' in history.history.keys():
            loss_label = 'Avg. Loss'
        else:
            loss_label = 'Single Loss'

        
    ax[-2].set_ylabel(loss_label)
    
    ax[-1].plot((1-np.array(history.history['exp_obs_ratio']))*100, 
              label='% Missing')
    ax[-1].plot(np.array(history.history['p_miss_row'])*100,
               label='% Missing rows')
    #ax[-1].set_ylabel('Fraction')
    ax[-1].set_xlabel('Training Epoch')
    ax[-1].legend()
    
    if bool_bivar:
        if 'corr' in history.history.keys():
            ax[0].plot(history.history['corr'])
            
        ax[0].set_xlabel('Training Epoch')
        ax[0].set_ylabel(r'$\hat{\Sigma}_{1,2}$')
        
    fig.set_size_inches( w = int(n_fig*5),h = 5)
    return fig    