# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 09:31:21 2018

This code file is for plotting the data of results_mat
Based on the data file in "Results_decision_parameter_june"
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def averaging_results(results_mat,times):
    #Averaging the result matrix table
    df = pd.DataFrame(columns = ['Decision parameter', 'Budget', 'AVG prob_initial(everybody)'
                                              ,'AVG prob_final(everybody)', 'AVG prob_initial(decided)','AVG prob_final(decided)'
                                              ,'AVG used budget', 'Prob not decided initial', 'Prob not decided final',
                                              'AVG Meet Ha','AVG Meet RsA','env_0_initial','env_0_final', 'converge time','capacity'])
    ind = 0
    for df_ind, i in enumerate(range(int(results_mat.shape[0]/times))):
        df.loc[df_ind] = np.mean(results_mat[ind:ind+4], axis = 0)
        ind+=times
    return df

def building_xtics(decision_paramater):
    xtics = []
    for ind, dec_para in enumerate(decision_paramater):
        if ind%5 == 0:
            xtics.append(round(dec_para,3))
    return xtics
def final_plots(df, xtics, correct_A, correct_R_S):
    colors = ["windows blue", "faded green"]
    fig, axes  =  plt.subplots(nrows = 2, ncols = 2, figsize = (12,12))
    axes[0, 0].set_title('Probability for optimal decision vs. D_p \n (budget =100)', fontweight='bold')
    axes[0, 0].plot(df['Decision parameter'],df['AVG prob_initial(everybody)'])
    axes[0, 0].plot(df['Decision parameter'],df['AVG prob_initial(decided)'])
    axes[0, 0].plot(df['Decision parameter'],df['AVG prob_final(everybody)'])
    axes[0, 0].plot(df['Decision parameter'],df['AVG prob_final(decided)'])
    axes[0, 0].set_xlabel('Decision parameter')
    axes[0, 0].set_ylabel('Probability of optimal decision')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend(['inital(everybody)','inital(decided)','final(everybody)', 'final(decided)'])
    
    axes[0, 1].set_title('AVG met Human Agent vs. D_p \n (budget =100)', fontweight='bold')
    axes[0, 1].plot(df['Decision parameter'],df['AVG Meet Ha'])
    axes[0, 1].set_xlabel('Decision parameter')
    axes[0, 1].set_ylabel('AVG Meet Human Agent')    
    axes[0, 1].set_ylim(0, df['AVG Meet Ha'].max())
    
    axes[1, 0].set_title('Decided (initial and final) vs. D_p \n (budget =100)', fontweight='bold')
    width_bar = df['Decision parameter'].max()/(df['Decision parameter'].shape[0] - 1)
    axes[1, 0].bar(df['Decision parameter'],height = (1 - df['Prob not decided final']), width = width_bar)
    axes[1, 0].bar(df['Decision parameter'],height = (1 - df['Prob not decided initial']), width = width_bar)
    axes[1, 0].set_xlabel('Decision parameter')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlim(0, df['Decision parameter'].max())
    axes[1, 0].legend(['final decided','inital decided'])  

    axes[1, 1].set_title('People in environment vs. D_p \n (budget =100)', fontweight='bold')
    axes[1, 1].plot(df['Decision parameter'],df['env_0_initial'], color =  sns.xkcd_rgb[colors[0]], dashes=[6, 2])
    axes[1, 1].plot(df['Decision parameter'],df['env_0_final'], color =  sns.xkcd_rgb[colors[0]])
    axes[1, 1].plot(df['Decision parameter'],df['Prob not decided initial'] - df['env_0_initial'], color =  sns.xkcd_rgb[colors[1]], dashes=[6, 2])
    axes[1, 1].plot(df['Decision parameter'],df['Prob not decided final'] - df['env_0_final'], color =  sns.xkcd_rgb[colors[1]])
    axes[1, 1].set_xlabel('Decision parameter')
    axes[1, 1].set_ylabel('Precent of the people')   
    axes[1, 1].legend(['inital 0 envi','final 0 envi','intial 2+ envi', 'final 2+ envi'])

def plot_finalprob_emptyspace(df, xtics):
    fig, axes  =  plt.subplots(nrows = 1, ncols = 3, figsize = (18,6))
    for ind, i in enumerate(df['Budget'].unique()):
        if (ind ==0) or (ind == 1) or (ind == 2):
            plt.sca(axes[0])
            plt.title('AVG delta final-initial prob(everybody) \n vs. decision paramater')
            plt.plot(df[df['Budget'] == i]['Decision parameter'],
                     df[df['Budget'] == i]['AVG prob_final(everybody)'] - df[df['Budget'] == i]['AVG prob_initial'])
            plt.xticks(xtics, rotation = 90)
            plt.legend(['Budget = ' + str(df['Budget'].unique()[0]), 'Budget = ' + str(df['Budget'].unique()[1]),
                        'Budget = ' + str(df['Budget'].unique()[2])])

    for ind, i in enumerate(df['Budget'].unique()):
        if (ind ==0) or (ind == 1) or (ind == 2):    
            plt.sca(axes[1])
            plt.title('AVG final sucess prob(decided) \n vs. decision paramater')
            plt.plot(df[df['Budget'] == i]['Decision parameter'],
                     df[df['Budget'] == i]['AVG prob_final(decided)'])
            plt.xticks(xtics, rotation = 90)
            plt.legend(['Budget = ' + str(df['Budget'].unique()[0]), 'Budget = ' + str(df['Budget'].unique()[1]),
                        'Budget = ' + str(df['Budget'].unique()[2])])
    for ind, i in enumerate(df['Budget'].unique()):
        if (ind ==0) or (ind == 1) or (ind == 2):    
            plt.sca(axes[2])
            plt.title('AVG prob not decided \n vs. decision paramater')
            plt.plot(df[df['Budget'] == i]['Decision parameter'],
                     df[df['Budget'] == i]['Prob not decided'])
            plt.xticks(xtics, rotation = 90)
            plt.legend(['Budget = ' + str(df['Budget'].unique()[0]), 'Budget = ' + str(df['Budget'].unique()[1]),
                        'Budget = ' + str(df['Budget'].unique()[2])])
    plt.show()
def plot_used_budget(df, xtics):
    #fig, ax = plt.subplot()
    plt.figure(figsize = (6,6))
    for ind, i in enumerate(df['Budget'].unique()):
        if (ind ==0) or (ind == 1) or (ind == 2):
            plt.title('AVG used budget \n vs. decision paramater')
            plt.plot(df[df['Budget'] == i]['Decision parameter'],
                     df[df['Budget'] == i]['AVG used budget'])
            plt.xticks(xtics, rotation = 90)
            plt.legend(['Budget = ' + str(df['Budget'].unique()[0]), 'Budget = ' + str(df['Budget'].unique()[1]),
                        'Budget = ' + str(df['Budget'].unique()[2])])

############################################################
############## Plotting for different Budget ###############
############################################################
def many_results_mat(results_mat_10, results_mat_25, results_mat_no_lim):
    l = [results_mat_10, results_mat_25, results_mat_no_lim]
    d_df={}    
    for x in range(3):
        d_df[x] = averaging_results(l[x])
    return d_df

    
    
def plot_decision_para_capacity(d_df, xtics):
    fig, ax1 = plt.subplots()
    colors = ["windows blue", "faded green", "dusty purple"]
    ax1.set_title('Budget = 100')
    ax1.set_xlabel('Decision parameter')
    ax1.set_ylabel('Final Probability of optimal decision')
    ax1.set_xlim(d_df[0][d_df[0]['Budget'] == 100]['Decision parameter'].iloc[0], d_df[0][d_df[0]['Budget'] == 100]['Decision parameter'].iloc[-1])
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(xtics, rotation = 90)
    for ind, i in enumerate(range(len(d_df))):
        ax1.plot(d_df[i][d_df[i]['Budget'] == 100]['Decision parameter'],
                 d_df[i][d_df[i]['Budget'] == 100]['AVG prob_final(decided)'], color =  sns.xkcd_rgb[colors[ind]])
        #ax1.xticks(xtics, rotation = 90)
    ax1.legend(['Capacity = 110%', 'Capacity = 125%',
                     'Unlimited Capacity'])
    ax2 = ax1.twinx()
    ax2.set_ylabel('People have decided (dashed)')
    ax2.set_ylim(0, 1.05)
    for ind, i in enumerate(range(len(d_df))):
        ax1.plot(d_df[i][d_df[i]['Budget'] == 100]['Decision parameter'].iloc[1:-1],
                 1 - d_df[i][d_df[i]['Budget'] == 100]['Prob not decided'].iloc[1:-1], color =  sns.xkcd_rgb[colors[ind]], dashes=[6, 2])
def heat_map(df, xtics):
    
    fig, axes  =  plt.subplots(nrows = 2, ncols = 2, figsize = (18,12)) #sharey=True, sharex=True
    
    Hm_final_not_dec = np.reshape(df['Prob not decided final'],(10,50))
    axes[0, 0].set_title('Heatmap Prob Decided \n Budget & D_p', fontweight='bold')
    sns.heatmap(1- Hm_final_not_dec,ax = axes[0, 0], cmap="viridis", vmin=0, vmax=1,
                xticklabels = False, cbar_kws={"orientation": "horizontal"})
    axes[0, 0].set_yticklabels(pd.unique(round(df['Budget'],1)), rotation = 0)
    axes[0, 0].set_xlabel('Decision parameter')
    axes[0, 0].set_ylabel('Budget')
    

    Hm_AVG_used_budget = np.reshape(df['AVG used budget'],(10,50))
    axes[0, 1].set_title('Heatmap AVG used Budget \n Budget & D_p', fontweight='bold')
    sns.heatmap(Hm_AVG_used_budget,ax = axes[0, 1], vmin=0, vmax=np.max(df['Budget']),
                xticklabels = False, cbar_kws={"orientation": "horizontal"})
    axes[0, 1].set_xlabel('Decision parameter')
    axes[0, 1].set_ylabel('Budget')   
    axes[0, 1].set_yticklabels(pd.unique(round(df['Budget'],1)), rotation = 0)
    
    Hm_0_Neighbour = np.reshape(df['env_0_final'],(10,50))
    axes[1, 0].set_title('Heatmap 0 in Neighbourhood \n Budget & D_p', fontweight='bold')
    sns.heatmap(Hm_0_Neighbour,ax = axes[1, 0], cmap="viridis", vmin=0, vmax=1,
                xticklabels = False, cbar=False)
    axes[1, 0].set_yticklabels(pd.unique(round(df['Budget'],1)), rotation = 0)
    axes[1, 0].set_xlabel('Decision parameter')
    axes[1, 0].set_ylabel('Budget')

    Hm_2plus_Neighbour = np.reshape(df['Prob not decided final'] - df['env_0_final'],(10,50))
    axes[1, 1].set_title('Heatmap 2+ in Neighbourhood \n Budget & D_p', fontweight='bold')
    sns.heatmap(Hm_2plus_Neighbour,ax = axes[1, 1], cmap="viridis", vmin=0, vmax=1,
                xticklabels = False, cbar=False)
    axes[1, 1].set_yticklabels(pd.unique(round(df['Budget'],1)), rotation = 0)
    axes[1, 1].set_xlabel('Decision parameter')
    axes[1, 1].set_ylabel('Budget')

def linear_fit_budger_2_Neighbourhood(df, xtics):
    lm = LinearRegression()
    Hm_2plus_Neighbour = np.reshape(df['Prob not decided final'] - df['env_0_final'],(10,50))
    
    X = np.vstack((np.ones(len(pd.unique(df['Decision parameter']))), pd.unique(df['Decision parameter']))).T
    y = np.mean(Hm_2plus_Neighbour, axis = 0)    
    
    lm.fit(X, y)
    y_pred = lm.predict(X)
    r_2 = r2_score(y, y_pred)
    
    fig, ax1 = plt.subplots()
    
    ax1.set_title('Prob of being in 2+ Neighbourhood vs. $D_p$ \n (AVG prices)')
    ax1.plot(pd.unique(df['Decision parameter']),y_pred)
    ax1.scatter(pd.unique(df['Decision parameter']),y, color= 'black', s= 6)
    ax1.tick_params(xtics, rotation = 90)
    ax1.set_xlabel('Decision parameter')
    ax1.set_ylabel('Final Prob of Being in 2+ Neighbourhood')
    ax1.legend(['linearReg \n $R^2$ = {0}'.format(round(r_2,2)), 'Data'])
df = averaging_results(results_mat,times)
xtics = building_xtics(decision_paramater)
#final_plots(df, xtics, correct_A, correct_R_S)
heat_map(df, xtics)
linear_fit_budger_2_Neighbourhood(df, xtics)
sns.set()
#d_df = many_results_mat(results_mat_10, results_mat_25, results_mat_no_lim)
#plot_decision_para_capacity(d_df, xtics)
#plot_finalprob_emptyspace(df, xtics)
#plot_used_budget(df, xtics)

