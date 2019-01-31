# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 14:23:11 2018

@author: Itay
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

def arrange_and_plots(results_mat, times):
    reg_mat = results_mat[:,(21, 0,5,9,12,13,14,15,16,17,18,19,20)]

    reg_df = pd.DataFrame(reg_mat,columns =['P_a','Decision parameter','AVG prob_final(decided)','SUM Meet Ha','Prob_WOM', 'Prob_WOM_Ha','Per_SH',
                          'Decided by Normal_WOM','Good dec by Normal_WOM','Decided by SH_WOM','Good dec by SH_WOM','Decided by HA','Good dec by HA'] )
    
    df = pd.DataFrame(columns =['P_a','Decision parameter','AVG prob_final(decided)','SUM Meet Ha','Prob_WOM', 'Prob_WOM_Ha','Per_SH',
                          'Decided by Normal_WOM','Good dec by Normal_WOM','Decided by SH_WOM','Good dec by SH_WOM','Decided by HA','Good dec by HA'])
    df_std = pd.DataFrame(columns =['P_a','Decision parameter','AVG prob_final(decided)','SUM Meet Ha','Prob_WOM', 'Prob_WOM_Ha','Per_SH',
                          'Decided by Normal_WOM','Good dec by Normal_WOM','Decided by SH_WOM','Good dec by SH_WOM','Decided by HA','Good dec by HA'])
    ind = 0
    for df_ind, i in enumerate(range(int(reg_df.shape[0]/(times)))):
        df.loc[df_ind] = reg_df.iloc[ind:ind+times].mean(axis = 0)
        df_std.loc[df_ind] = reg_df.iloc[ind:ind+times].std(axis = 0)
        ind+=times
    
    df['Re_N_WOM'] = df['Good dec by Normal_WOM']/ df['Decided by Normal_WOM']
    df['Re_SH_WOM'] = df['Good dec by SH_WOM']/df['Decided by SH_WOM']
    df['Re_Ha'] = df['Good dec by HA']/df['Decided by HA']
    return df, df_std    

def Plots_Pa(df, df_std):
    ''' Plots of the factor that help to make the decision, which meand the last encounter
    and afterwards the decision has made 
    
    
    Can be seen that P_a can help to minimize the impact of the Social hubs, but not really
    generating an effect that helps the network to make better decisions'''
    
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (15,6))
    columns_groups =['P_a','Prob_WOM', 'Prob_WOM_Ha']
    columns_graphs = ['Decided by Normal_WOM','Good dec by Normal_WOM','Decided by SH_WOM','Good dec by SH_WOM',
                      'Decided by HA','Good dec by HA']
    colors = ['navy','navy','seagreen','seagreen','crimson','crimson']
    fig.suptitle('No. Decdied and No. Optimal Decided \n vs. Probabilites for Interactions')
    label_x = ['$P_{HA}$','$P_{WOM_d}$','$P_{WOM_a}$']
    for ind, i in enumerate(columns_groups):
        d = df.groupby(i, axis = 0).mean()       
        for ind_j, j in enumerate(columns_graphs):
            if ind_j %2 == 0:
                axes[ind].plot(d.index, d[j],'-o', color = colors[ind_j], markersize=6, linewidth = 1)
            else:
                axes[ind].plot(d.index, d[j],'--*', color = colors[ind_j], markersize=6, linewidth = 1)
        axes[ind].set_ylim(0, 320)    
        axes[ind].set_ylabel('Number of People')   
        axes[ind].set_xlabel(label_x[ind])  
        axes[ind].legend(['Proximity WOM','Optimal Proximity WOM','Social Hub WOM','Optimal Social Hub WOM',
                      'Human Agent','Optimal Human Agent'], frameon = True)
            #axes[ind].set_title(i + ' vs. AVG prob_final' , fontweight='bold')           

    fig2, axes2 = plt.subplots(nrows = 1, ncols = 3, figsize = (15,6))
    columns_groups =['P_a','Prob_WOM', 'Prob_WOM_Ha']
    columns_graphs = ['Re_N_WOM','Re_SH_WOM','Re_Ha']
    label_x = ['$P_{HA}$','$P_{WOM_d}$','$P_{WOM_a}$']
    fig2.suptitle('Relation of No. Optimal Decided/No. Decided \n vs. Probabilites for Interactions')
    for ind, i in enumerate(columns_groups):
        d = df.groupby(i, axis = 0).mean()       
        for ind_j, j in enumerate(columns_graphs):
            axes2[ind].plot(d.index, d[j],'-o', markersize=6, linewidth = 1, color = colors[2*ind_j])
        axes2[ind].set_ylim(0.3, 1)    
        axes2[ind].set_ylabel('Relation (Optimal Decided)/Decided')   
        axes2[ind].set_xlabel(label_x[ind])  
        axes2[ind].legend(['Proximity WOM','Social Hub WOM','Human Agent'], frameon = True)


    
def clean_before_regression(results_mat, times):
    ' ' ' Here I had to correct the number of rows and to insert column of P_a '''
    reg_mat = results_mat[:,(21,0,12,13,14,5)]

    reg_df = pd.DataFrame(reg_mat,columns =['P_a','Decision parameter','Prob_WOM', 'Prob_WOM_Ha','Per_SH','AVG prob_final(decided)'] )
    
    df = pd.DataFrame(columns =['P_a','Decision parameter','Prob_WOM', 'Prob_WOM_Ha','Per_SH','AVG prob_final(decided)'])
    df_std = pd.DataFrame(columns =['P_a','Decision parameter','Prob_WOM', 'Prob_WOM_Ha','Per_SH','AVG prob_final(decided)'])
    ind = 0
    for df_ind, i in enumerate(range(int(reg_df.shape[0]/(times)))):
        df.loc[df_ind] = reg_df.iloc[ind:ind+times].mean(axis = 0)
        df_std.loc[df_ind] = reg_df.iloc[ind:ind+times].std(axis = 0)
        ind+=times
    return reg_df, df, df_std                

def OLS(df):
    Y = df['AVG prob_final(decided)']
    X = df[['P_a','Decision parameter','Prob_WOM', 'Prob_WOM_Ha','Per_SH']]
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results_ols = model.fit()
    return results_ols

def Plots_parameters(df):
    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (18,12))
    #Y = df['AVG prob_final(decided)']
    
    X = df[['P_a','Decision parameter','Prob_WOM', 'Prob_WOM_Ha','Per_SH']]
    for ind, i in enumerate(X.columns):
        d = df.groupby(i, axis = 0).mean()    
        d_std = df.groupby(i, axis = 0).std()    
        if ind <3:
            axes[0,ind].plot(d.index, d['AVG prob_final(decided)'],'--*')
            axes[0,ind].set_title(i + ' vs. AVG prob_final' , fontweight='bold')
            axes[0,ind].set_ylim(0.5, 1)
            axes[0,ind].errorbar(d.index,d['AVG prob_final(decided)'],d_std['AVG prob_final(decided)'], color = sns.xkcd_rgb["windows blue"])
        else:
            axes[1,ind-3].plot(d.index, d['AVG prob_final(decided)'],'--*')
            axes[1,ind-3].set_title(i + ' vs. AVG prob_final' , fontweight='bold')
            axes[1,ind-3].set_ylim(0.5, 1)
            axes[1,ind-3].errorbar(d.index,d['AVG prob_final(decided)'],d_std['AVG prob_final(decided)'], color = sns.xkcd_rgb["windows blue"])
        
    ## Plot for different D_p
    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (12,12))
    unique_Dp = X['Decision parameter'].unique()
    for ind, i in enumerate(X.columns[([0,2,3,4])]):           
        if ind <2:
            for Dp in unique_Dp:
                d = df[df['Decision parameter']==Dp].groupby(i, axis = 0).mean()
                d_std = df[df['Decision parameter']==Dp].groupby(i, axis = 0).std()
                axes[0,ind].plot(d.index, d['AVG prob_final(decided)'],'--*')
                axes[0,ind].set_title(i + ' vs. AVG prob_final' , fontweight='bold')
                axes[0,ind].set_ylim(0.5, 1)
            axes[0,ind].legend(np.round(unique_Dp,3), frameon = True)
                #axes[0,ind].errorbar(d.index,d['AVG prob_final(decided)'],d_std['AVG prob_final(decided)'], color = sns.xkcd_rgb["windows blue"])
        else:
            for Dp in unique_Dp:
                d = df[df['Decision parameter']==Dp].groupby(i, axis = 0).mean()
                d_std = df[df['Decision parameter']==Dp].groupby(i, axis = 0).std()
                axes[1,ind-2].plot(d.index, d['AVG prob_final(decided)'],'--*')
                axes[1,ind-2].set_title(i + ' vs. AVG prob_final' , fontweight='bold')
                axes[1,ind-2].set_ylim(0.5, 1)
                

def OLS_summary_as_csv(results_ols):
    beginningtex = """\\documentclass{report}
    \\usepackage{booktabs}
    \\begin{document}"""
    endtex = "\end{document}"
    f = open('myreg.csv', 'w')
    f.write(beginningtex)
    f.write(results_ols.summary().as_csv())
    f.write(endtex)
    f.close()

def Horizontal_bat_plot_OLS(results_ols):
    
    fig, ax = plt.subplots()
    
    colors = ['#9B0029' if x < 0 else '#003366' for x in results_ols.params]
    y_pos = np.arange(len(results_ols.params.index))
    ylabel = ['const', '$P_{HA}$', '$D_p$', '$P_{WOM_a}$', '$P_{WOM_d}$','$Per_{SH}$']
    
    ax.barh(y_pos, results_ols.params, align='center',
            color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ylabel)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Coefficient values')
    ax.set_xlim([-np.max(np.abs(results_ols.params)*1.1),np.max(np.abs(results_ols.params)*1.1)])
    ax.set_title('OLS Parameters values \n  $R^2$ = ' +  str(np.round(results_ols.rsquared,3)), fontweight='bold')

def SH_vs_Pa(df):
    fig, ax = plt.subplots()
    #x = df['P_a'][df['Decision parameter']==unique_Dp[1]]
    #y = df['Per_SH'][df['Decision parameter']==unique_Dp[1]]
    
    values = np.zeros((len(df['Per_SH'].unique()),len(df['P_a'].unique())))
    values_std = np.zeros((len(df['Per_SH'].unique()),len(df['P_a'].unique())))
    #values = np.random.rand(len(df['P_a'].unique()),len(df['P_a'].unique()))
    for ind_x, i in enumerate(df['P_a'].unique()):
        for ind_y, j in enumerate(df['Per_SH'].unique()):
            values[ind_x,ind_y ] = df[(df['P_a'] == i) & (df['Per_SH'] == j)].mean().loc['AVG prob_final(decided)']     # & (df['Decision parameter']==unique_Dp[1])
            values_std[ind_x,ind_y ] = df[(df['P_a'] == i) & (df['Per_SH'] == j)].std().loc['AVG prob_final(decided)']
#            values[ind_x, ind_y] = df['AVG prob_final(decided)'][df['Decision parameter']==unique_Dp[1]]
    im = ax.imshow(values,  cmap=plt.get_cmap("summer", 3))
    ax.set_xticks(np.arange(len(df['Per_SH'].unique())))
    ax.set_yticks(np.arange(len(df['P_a'].unique())))
    
    ax.set_yticklabels(np.round(df['P_a'].unique(),3))
    ax.set_xticklabels(np.round(df['Per_SH'].unique(),3))
    for i in range(len(df['P_a'].unique())):
        for j in range(len(df['Per_SH'].unique())):
            text = ax.text(j, i, np.round(values[i, j],2),
                           ha="center", va="center", color="k")
    ax.set_title("Relation of $P_{HA}$ and $Per_{SH}$ \n Prob. for Optimal Decision")  
    ax.set_ylabel('$P_{HA}$')  
    ax.set_xlabel('$Per_{SH}$')
    plt.colorbar(im)
    ax.grid(False)
    
    return values, values_std    
    #fig, ax2 = plt.subplots()
    #ax2.plt.scatter(df['P_a'][df['Decision parameter']==unique_Dp[1]], df['Per_SH'][df['Decision parameter']==unique_Dp[1]], c =  df['AVG prob_final(decided)'][df['Decision parameter']==unique_Dp[1]])
    

sns.set()
reg_df, df, df_std = clean_before_regression(results_mat, times)
results_ols = OLS(df)
#OLS_summary_as_csv(results_ols)
Horizontal_bat_plot_OLS(results_ols)
print(results_ols.summary())

df_plots, df_plots_std = arrange_and_plots(results_mat, times)
Plots_Pa(df_plots, df_plots_std)

Plots_parameters(df)

values, values_std = SH_vs_Pa(df)
