from utilities import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from matplotlib.cm import ScalarMappable
from random import randrange
import matplotlib.cm as cm


import plotly.express as px

def get_simple_parcoords(df, save_title, color = None,color_scale = 'tealgrn'):
    if color == 'None':
        color = 'flood_risk'
    fig = px.parallel_coordinates(df, color= color, dimensions = ['hydropower_revenue', 'baltimore_discharge', 'environment','recreation',  'atomic_power_plant_discharge',
            'chester_discharge', 
        'flood_risk'], color_continuous_scale=color_scale , labels = {'hydropower_revenue': 'Hydropower', 'baltimore_discharge' : 'Baltimore', 'environment' : 'Environment','recreation' : 'Recreation',  'atomic_power_plant_discharge' :' Atomic Power',
            'chester_discharge' : 'Chester', 
        'flood_risk' : 'Flood Risk'})
    fig.update_layout(width = 1000, height = 400, font_family = "Georgia", font_size = 12,
    )
    fig.show()
    fig.write_image(f'../visuals/{save_title}.png')


import plotly.graph_objects as go

def plot_selected_parcoords(selected_payoffs, range_df, save_title, colorscale='viridis', color_column = None):

    if color_column == None:
        color = np.array(list(range(0,selected_payoffs.shape[0])))
    else:
        color = selected_payoffs[color_column]

    tick_values = [x*1/(selected_payoffs.shape[0]-1) for x in range(selected_payoffs.shape[0])]
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = color, colorscale = colorscale#colors#['#e6194b', '#46f0f0', '#ffe119', '#4363d8', '#f58231', '#911eb4']
            #['#003f5c','#2f4b7c','#665191','#d45087','#f95d6a', '#ff7c43','#ffa600', '#a05195'] 
            ),
            dimensions = list([
                dict(range =[0,1],
                    #constraintrange = [1,2], # change this range by dragging the pink line
                    label = 'Policy', values = tick_values, tickvals = tick_values,
                    ticktext = list(selected_payoffs.index) ),

                dict(range =[min(range_df['hydropower_revenue']), max(range_df['hydropower_revenue'])],
                    #range = [0,1],    
                    #constraintrange = [0,0.5], 
                    label = 'Hydropower Revenue', values = selected_payoffs['hydropower_revenue']),

                dict(range =[min(range_df['baltimore_discharge']), max(range_df['baltimore_discharge'])],#range = [0,1],    
                    #constraintrange = [0,0.5], 
                    label = 'Baltimore', values = selected_payoffs['baltimore_discharge']),

                dict(range =[min(range_df['environment']), max(range_df['environment'])],#range = [0,1],    
                    #constraintrange = [0,0.5], 
                    label = 'Environment', values = selected_payoffs['environment']),

                dict(range =[min(range_df['atomic_power_plant_discharge']), max(range_df['atomic_power_plant_discharge'])],#range = [0,1],    
                    #constraintrange = [0,0.5], 
                    label = 'Atomic Power', values = selected_payoffs['atomic_power_plant_discharge']),

                dict(range =[min(range_df['recreation']), max(range_df['recreation'])],#range = [0,1],    
                    #constraintrange = [0,0.5], 
                    label = 'Recreation', values = selected_payoffs['recreation']),

                dict(range =[min(range_df['chester_discharge']), max(range_df['chester_discharge'])],#range = [0,1],
                    #constraintrange = [1,2], # change this range by dragging the pink line
                    label = 'Chester', values = selected_payoffs['chester_discharge']),

                dict(range =[min(range_df['flood_risk']), max(range_df['flood_risk'])],#range = [0,1],    
                    #constraintrange = [0,0.5], 
                    label = 'Flood Risk', values = selected_payoffs['flood_risk'])
                
            ])
        )
    )
    fig.update_layout(width = 1000, height = 400, font_family = "Georgia", font_size = 14,
    #title = {'text': "Visualizing tradeoffs among fair policies", 'xanchor': 'center', 'yanchor': 'top', 'x': 0.4, 'y': 0.10}
    )
    fig.write_image(f'../visuals/{save_title}.png')

    fig.show()

def plot_selected_parcoords_without_titles(selected_payoffs, range_df, save_title, colorscale='viridis', color_column = 'gini_index'):

    tick_values = [x*1/(selected_payoffs.shape[0]-1) for x in range(selected_payoffs.shape[0])]
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = selected_payoffs[color_column], colorscale = colorscale, showscale = True, 
            ),
            dimensions = list([

                dict(range =[min(range_df['hydropower_revenue']), max(range_df['hydropower_revenue'])],
                    label = 'Hydropower Revenue', values = selected_payoffs['hydropower_revenue']),

                dict(range =[min(range_df['baltimore_discharge']), max(range_df['baltimore_discharge'])],
                    label = 'Baltimore', values = selected_payoffs['baltimore_discharge']),

                dict(range =[min(range_df['environment']), max(range_df['environment'])],
                    label = 'Environment', values = selected_payoffs['environment']),

                dict(range =[min(range_df['atomic_power_plant_discharge']), max(range_df['atomic_power_plant_discharge'])],#
                    label = 'Atomic Power', values = selected_payoffs['atomic_power_plant_discharge']),

                dict(range =[min(range_df['recreation']), max(range_df['recreation'])],
                    label = 'Recreation', values = selected_payoffs['recreation']),

                dict(range =[min(range_df['chester_discharge']), max(range_df['chester_discharge'])],
                    label = 'Chester', values = selected_payoffs['chester_discharge']),

                dict(range =[min(range_df['flood_risk']), max(range_df['flood_risk'])],
                    label = 'Flood Risk', values = selected_payoffs['flood_risk'])
                
            ])
        )
    )
    fig.update_coloraxes(showscale = True)

    fig.update_layout(width = 1000, height = 400, font_family = "Georgia", font_size = 14,
    #title = {'text': "Visualizing tradeoffs among fair policies", 'xanchor': 'center', 'yanchor': 'top', 'x': 0.4, 'y': 0.10}
    )
    fig.write_image(f'../visuals/{save_title}.png')

    fig.show()

def get_utility_equality_scatter(df, xmetric, ymetric, save_title):

    if xmetric == 'gini_index':
        xlabel = 'gini index'
    elif xmetric == 'prioritarian_welfare':
        xlabel = 'prioritarian utility'
    else:
        xlabel = xmetric

    if ymetric == 'gini_index':
        ylabel = 'gini index'
    elif ymetric == 'prioritarian_welfare':
        ylabel = 'prioritarian utility'
    else:
        ylabel = ymetric

    # define figure and numebr of subplots
    fig = plt.figure(figsize = (12,6))
    ax = fig.subplots(nrows = 2, ncols = 4, sharex = True, sharey = True )

    actor_names = ['Hydropower Revenue', 'Baltimore', 'Environment', 'Atomic Power', 'Recreation', 'Chester', 'Flood Risk' ]
    actors = ['hydropower_revenue', 'baltimore_discharge','environment','atomic_power_plant_discharge','recreation','chester_discharge', 'flood_risk']

    # color and color scale legend
    scales = np.linspace(0,1)
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(scales.min(), scales.max())

    # loop to plot in different subplots
    for idx, actor in enumerate(actors):

        # id of each subplot
        axes = ax.flatten()
        
        # scatter plot which can be changed to any other plot
        g = sns.scatterplot(data = df, x=xmetric, y=ymetric, hue = actor, ax = axes[idx],  s=25, palette = 'viridis', vmin = 0, vmax = 1, hue_order = range(0,1))

        # legend
        axes[idx].get_legend().remove()

        # titles and labels for individual subplots
        axes[idx].set_title(actor_names[idx], font = 'Georgia', fontsize = 14)
        axes[idx].set_ylabel(ylabel, font = 'Georgia', fontsize = 12)
        axes[idx].set_xlabel(xlabel, font = 'Georgia', fontsize = 12)

        #axes lines - top, right, bottom and left
        axes[idx].spines["top"].set_visible(False)
        axes[idx].spines["right"].set_visible(False)
        axes[7].axis('off')

        #ticks
        axes[idx].tick_params(axis=u'both', which=u'both',length=0)

    sm = ScalarMappable(norm = norm, cmap = cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax = axes[7])
    cbar.ax.set_title("actor utility", font = 'Georgia', fontsize = 12)

    plt.savefig(f'../visuals/{save_title}.png')

def plot_outcome_utility(utility_function = None, payoffs = None, label = 'compare utilities', rho = None):

    fig = plt.figure(figsize = (18,5))
    ax = fig.subplots(nrows = 2, ncols = 4)
    #payoffs[['environment', 'flood_risk']] = payoffs[['environment', 'flood_risk']]*-1

    actor_names = ['Hydropower Revenue', 'Baltimore', 'Environment', 'Atomic Power', 'Recreation', 'Chester', 'Flood Risk' ]
    actors = ['hydropower_revenue', 'baltimore_discharge','environment','atomic_power_plant_discharge','recreation','chester_discharge', 'flood_risk']

    for idx, actor in enumerate(actors):
        axes = ax.flatten()

        if label == 'compare utilities':

            df = get_marginal_utility_of_df(payoffs, withNormalization=True, rho = 0)
            marginal = get_marginal_utility_of_df(payoffs, withNormalization=True, rho = 1.2)

            scatter = sns.scatterplot(x=payoffs[actor], y=df[actor], ax = axes[idx], label = 'linear utility', legend = False)
            scatter = sns.scatterplot(x = payoffs[actor], y = marginal[actor], ax = axes[idx], label = 'diminishing utility', legend = False)

            if idx == 6:
                scatter.legend(loc = 'center right', bbox_to_anchor= (1.2, 0.3), fontsize = 13, frameon = False )    

            legend_title = " "

        if label == 'linear utility':

            df = (utility_function(payoffs))
            df = df.sort_values(by = actor)
            payoffs = payoffs.sort_values(by = actor)

            sns.scatterplot(x=payoffs[actor], y=df[actor], ax = axes[idx])

        if label == 'diminishing utility' and rho == None:

            

            for rhos in np.arange(0, 2.5, 1).tolist():
                
                df = (utility_function(payoffs, rhos))
                df = df.sort_values(by = actor, ascending = True)
                payoffs = payoffs.sort_values(by = actor)

                if actor in ['environment', 'flood_risk']:
                    df[actor] = df[actor]*-1
                    df1 = normalize(df)
                #     df = df.sort_values(by = actor, ascending = False)
                #     payoffs = payoffs.sort_values(by = actor, ascending = True)
                else:
                    df1 = normalize(df)
                
    

                axes[idx].plot(payoffs[actor],df1[actor], label = rhos, color = cm.viridis((rhos/2.5)))    

            legend_title = "marginal utility of consumption"        

        elif label == 'diminishing utility' and rho == 'default':

            payoffs = payoffs.sort_values(by = actor)

            df = (utility_function(payoffs))
            df = df.sort_values(by = actor)

            axes[idx].plot(payoffs[actor],df[actor])
        
        
        # titles and labels
        axes[idx].set_title(actor_names[idx], font = 'Georgia', fontsize = 14)
        axes[idx].set_ylabel('utility', font = 'Georgia', fontsize = 12)
        axes[idx].set_xlabel('outcome', font = 'Georgia', fontsize = 12)

        #axes lines
        axes[idx].spines["top"].set_visible(False)
        axes[idx].spines["right"].set_visible(False)
        axes[7].axis('off')

        #ticks
        axes[idx].tick_params(axis=u'x', which=u'both',length=0, labelbottom = True)
    
    # legend
    if idx < 6:
        axes[idx].get_legend().remove()
    axes[6].legend(frameon = False, loc = 'lower right', bbox_to_anchor = (1.5, 0.1), 
    title = legend_title, prop = 'Georgia', borderaxespad = 2, columnspacing = 1, mode = "expand")
    
    #plt.title(f"{label}")

    fig.tight_layout()
    plt.savefig(f'../visuals/{label}.png')

def get_swarmplot(df, baselines, save_title):

    sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)

    fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (12,8), sharex = True)

    for idx, baseline in enumerate(baselines):

        df1 = df[df['baseline']==baseline]
        df1 = df1.sort_values(by = ['color'])

        axes = ax.flatten()    
        sns.swarmplot(data = df1, x = 'resultant', y = 'baseline', size = 8, ax = axes[idx], hue ='color', palette = ['#440154','darkred','#2D708E','#f2f2f2'])

        # titles and labels
        #axes[idx].set_title(actor_names[idx], font = 'Georgia', fontsize = 14)
        #axes[idx].set_ylabel('', font = 'Georgia', fontsize = 12)
        axes[0].set_ylabel('Susquehanna status quo', font = 'Georgia', fontsize = 12, rotation = 'horizontal')
        axes[1].set_ylabel('Get nothing           ', font = 'Georgia', fontsize = 12, rotation = 'horizontal')
        axes[2].set_ylabel('No disagreement point', font = 'Georgia', fontsize = 12, rotation = 'horizontal')
        axes[0].set_xlabel('', font = 'Georgia', fontsize = 12)
        axes[1].set_xlabel('', font = 'Georgia', fontsize = 12)
        axes[2].set_xlabel('preference from fallback bargaining', font = 'Georgia', fontsize = 12)

        #axes lines
        axes[idx].spines["top"].set_visible(False)
        axes[idx].spines["right"].set_visible(False)
        axes[idx].spines["left"].set_visible(False)
        axes[0].spines['bottom'].set_visible(False)
        axes[1].spines['bottom'].set_visible(False)

        #ticks
        axes[idx].tick_params(axis=u'both', which=u'both',length=0)
        axes[idx].tick_params(axis=u'y', which=u'major',length=0, labelsize =0)

        #legend
        axes[idx].get_legend().remove()

        for x, policy in zip(df1['resultant'], df1['decision']):
            if x>=np.percentile(df1['resultant'], 98.5) or x<np.percentile(df1['resultant'], 1) or policy == 'baseline':
                if policy == 'baseline' and baseline == 'Status Quo':
                    y_loc = 0.2
                else:
                    y_loc = -0.14
                if randrange(2) ==0:
                    v_al = 'top'
                    h_al = 'center'
                else:
                    if randrange(2) == 1:
                        v_al = 'bottom'
                        h_al = 'right'
                    else:
                        v_al = 'center_baseline'
                        h_al = 'left'
                if policy == 'baseline':
                    policy1 = 'disagreement point'
                else:
                    policy1 = policy
                #  s= text to print
                axes[idx].text(x = x+0.4, y = y_loc, s = policy1, font = 'Georgia', horizontalalignment = h_al, va = v_al )

    #plt.title("Gini index of stable outcomes from fallback bargaining")

    #plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
    lgd = plt.legend(frameon = False, loc = 'upper right', bbox_to_anchor = (1.5, 1.5), 
    prop = 'Georgia', borderaxespad = 0.5)
    plt.savefig(f'../visuals/{save_title}.png', bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_gamma_values(welfare_function, payoffs, utility_function, gamma = None, rho = None):


    fig = plt.figure(figsize = (6,4))
    ax = fig.subplots(nrows = 1, ncols = 1)

    for gamma in np.arange(0, 3, 1).tolist():
        
        df = welfare_function(payoffs, utility_function, gamma, rho)
        df = normalize(df).sort_values(by = 'utility')
        #ax.plot(df['utility'],df['prioritarian_welfare'], label = gamma, color = cm.viridis((gamma/20)) )
        sns.scatterplot(data = df, x = 'utility', y = 'prioritarian_welfare', label = gamma, color = cm.viridis(gamma/2))

# legend
        ax.legend(frameon = False, loc = 'lower right', bbox_to_anchor = (1.5, 0.1), 
    title = 'degree of priority to worse-off', prop = 'Georgia', borderaxespad = 0.5)

    # titles and labels
    #ax.set_title(actor_names[idx], font = 'Georgia', fontsize = 14)
    ax.set_ylabel('transformed utility', font = 'Georgia', fontsize = 12)
    ax.set_xlabel('utility', font = 'Georgia', fontsize = 12)

    #axes lines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    #axes[7].axis('off')

    #ticks
    ax.tick_params(axis=u'both', which=u'both',length=0, labelbottom = True)

    fig.tight_layout()
    plt.savefig(f'../visuals/prioritarian_utility_vs_utility.png')

        





