import pandas as pd 
import os
from sklearn import preprocessing
import numpy as np
import timeit
import ast 
import networkx as nx
from matplotlib.pyplot import figure, text
import matplotlib.pyplot as plt
import math

class bargainingGames:
    '''Uses to create a bargaining game. 
    Specify payoffs, desired actors and desired aggregation method - resultant, cardinal_resultant, total_utility
    Steps - 
    1. Normalizes the payoffs
    2. Creates adjacency matrix 
    3. Plots each actors preference direction
    4. Calculates preference aggregation across all actors based on the desired method
    '''
    def __init__(self, payoffs, desired_actors):
        self.payoffs = payoffs
        self.desired_actors = desired_actors

    def set_parameters(self):

        decision_names = self.payoffs.T.columns
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(self.payoffs)
        #x_scaled
        normalized_payoffs = pd.DataFrame(x_scaled).reset_index(drop = True)
        normalized_payoffs.columns = list(self.payoffs.columns)
        normalized_payoffs.index = decision_names 
        normalized_payoffs.reset_index(inplace=True)
        normalized_payoffs.columns = ['decision'] + list(self.payoffs.columns)
        normalized_payoffs.reset_index(inplace=True)

        self.normalized_payoffs = normalized_payoffs

        self.decision_names = decision_names
        self.strategies = self.normalized_payoffs.T.drop('index')
        self.strategies.columns = list(self.strategies.iloc[0])
        self.strategies.drop('decision', inplace=True)
    
        self.adj_matrix = pd.crosstab(self.strategies.T.index, self.strategies.T.index).reset_index()
        self.adj_matrix.index = list(self.adj_matrix['row_0'])
        self.adj_matrix.drop(columns = 'row_0', inplace = True)
        for col in self.adj_matrix.columns:
            self.adj_matrix[col].values[:] = 0

        self.positions = {'policy7': ([ 0.13909726, -2]), 'policy15': ([1.63784725, 1.10417117]), 'policy16': ([-0.13340775,  1.95691638]), 'policy28': ([-1.60533389, -1.08271161]), 'policy30': ([ 1.75978156, -0.85477996]), 'policy46': ([-1.79798442,  0.87640402])}
        {'policy1': ([-0.57368514,  1.29224075]), 'policy2': ([-0.73785789, -0.1574678 ]), 'policy3': ([-2.        ,  0.26808907]), 'policy4': ([ 1.99370153, -0.21530045]), 'policy5': ([0.70991588, 1.34538909]), 'policy6': ([-0.66858356, -1.37803322])}
        self.fixed_nodes = self.positions.keys()

        


    def preference_direction(self, actor):
        strategies = self.strategies
        adj_matrix = self.adj_matrix
     

        G = nx.DiGraph()

        for index, values in strategies.items():
            G.add_node(index)
            G.nodes[index]['total_utility'] = 0
            for itemName, itemContent in values.items():
                for i in actor:
                    if itemName == i:
                        G.nodes[index][itemName] = itemContent
                        G.nodes[index]['total_utility'] += itemContent


        for index0 in actor:
            for index1 in adj_matrix:
                for index2 in strategies:
                    difference = strategies[index1][index0] - strategies[index2][index0]
                    if (difference) > 0.0:
                        #print(difference, " difference", index1, "index1", index2, "index2")
                        #adj_matrix[index2][index1] = adj_matrix[index2][index1] + 0.5
                        try:
                            G.edges[(index2, index1)]['weight'] = G.edges[(index2, index1)]['weight'] + 1
                            G.edges[(index2, index1)]['cardinal_weight'] = G.edges[(index2, index1)]['cardinal_weight'] + difference
                        except:
                            G.add_edge(index2, index1, weight=1)
                            G.add_edge(index2, index1, cardinal_weight = difference)

        

        for node in G.nodes():
            node_in_weight = 0
            node_out_weight = 0
            node_in_cardinal_weight = 0
            node_out_cardinal_weight = 0
            for i in G.in_edges(node):
                node_in_weight += G.edges[i]['weight']
                node_in_cardinal_weight += G.edges[i]['cardinal_weight']
            for i in G.out_edges(node):
                node_out_weight += G.edges[i]['weight']
                node_out_cardinal_weight += G.edges[i]['cardinal_weight']
            G.nodes[node]['resultant'] = node_in_weight - node_out_weight
            G.nodes[node]['cardinal_resultant'] = node_in_cardinal_weight - node_out_cardinal_weight
            G.nodes[node]['in_arrows'] = node_in_weight
            G.nodes[node]['out_arrows'] = node_out_weight

        return G   

    def plot_aggregation(self, graphs, criteria, actors, fixedPositions=False):

        columns = int(math.ceil(len(actors)/2))
        rows = int(math.ceil((len(actors))/columns))
        #if rows*columns < len(actors):
         #   rows = rows + 1
        #print(actors)
        #print(rows)
        #print(columns)

        fig, axes = plt.subplots(rows,columns, figsize = (20,10))
        actors = actors

        if fixedPositions:
            positions = self.positions
            fixed_nodes = self.fixed_nodes

        for i in range(len(graphs)):
            
            #print("i ", i)
            actor = actors[i]
            
            ax = axes.flatten()
            ax1 = ax[i]
    
            G = graphs[i]

            highest_preference = max([G.nodes[n][criteria] 
                                    for n in G.nodes()]) 

            lowest_preference = min([G.nodes[n][criteria] 
                                    for n in G.nodes()]) 

            labeldict = {}
            node_colors = []
            node_alpha = []
            for node in G.nodes():
                #labeldict[node] = node #+ ' ' + str(int(G.nodes[node]['resultant']))
                if G.nodes[node][criteria] == highest_preference:
                    node_colors.append("darkgreen")
                    node_alpha.append(1)
                elif G.nodes[node][criteria] == lowest_preference:
                    node_colors.append("darkred")
                    node_alpha.append(1)
                else:
                    node_colors.append("darkblue")
                    node_alpha.append(0.8)

            if fixedPositions:
                pos = nx.spring_layout(G, k = 0.5, weight = 0, scale = 2, pos = positions, fixed = fixed_nodes)
            else:
                pos = nx.spring_layout(G, k = 0.5, weight = 0, scale = 2)
            #print(pos)
        

            longest_path = nx.dag_longest_path(G)

            longest_edges = []
            edge_colors = []
            edge_alpha = []

            for edge in G.edges():
                for i in range(len(longest_path)-1):
                    if edge[0] == longest_path[i] and edge[1] == longest_path[i+1]:
                        longest_edges.append(edge)
                        #edge_colors.append("red")
                    #else:
                    # edge_colors.append("grey")
            
            #print(longest_edges)
            
            for edge in G.edges():
                if edge in longest_edges:
                    edge_colors.append("darkred")
                    edge_alpha.append(1)
                else:
                    edge_colors.append("grey")
                    edge_alpha.append(0.4)
            


            nx.draw_networkx_nodes(G, pos,  node_size = 200, node_color = node_colors, alpha = node_alpha, ax = ax1)

            # for node, (x, y) in pos.items():
            #     if G.nodes[node]['resultant'] == highest_preference:
            #         text(x, y,' '+ node + ' ', ha='left', va='bottom', fontsize = 11, fontname = "Georgia")

            for node in G.nodes():
                # if G.nodes[node]['resultant'] == highest_preference or G.nodes[node]['resultant'] == lowest_preference:
                labeldict[node] = node

            nx.draw_networkx_labels(G, pos, labels = labeldict, ax = ax1
            , horizontalalignment = 'center'
            ,verticalalignment = 'bottom',
            font_family = 'Georgia' )

            #print(len(edge_colors))
            #print(len(edge_alpha))

            edge_colors = np.array(edge_colors, dtype = "object")
            edge_alpha = np.array(edge_alpha, dtype = "object")

            nx.draw_networkx_edges(G, pos , arrows=True, 
                arrowsize=12,
                edge_color= edge_colors, #"grey",
                #alpha = edge_alpha, #0.8,
                #edge_cmap=plt.cm.Blues,
                width=2,
                ax = ax1)


            #nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
            #nx.draw_networkx_edges(G, pos,  arrows=False)

            ax = plt.gca()
            ax.set_axis_off()
            ax1.set_title(actor, font = 'Georgia', fontsize = 13)
            ax1.axis('off')
            #ax.title(actor)
        
        #return fig

        #plt.savefig(output_path, format = "PNG")
        plt.show(block = False)

    def combined_graph(self, all_outcome, criteria):

        positions = self.positions
        fixed_nodes = self.fixed_nodes

        edge_colors = "grey"
        G = all_outcome

        highest_preference = max([G.nodes[n][criteria] 
                                for n in G.nodes()]) 
        labeldict = {}
        node_colors = []
        node_alpha = []

        for node in G.nodes():
            #labeldict[node] = node #+ ' ' + str(int(G.nodes[node]['resultant']))
            if G.nodes[node][criteria] == highest_preference:
                node_colors.append("darkgreen")
                node_alpha.append(1)
            else:
                node_colors.append("darkblue")
                node_alpha.append(0.8)

        pos = nx.spring_layout(G, k = 0.5, weight = 0, scale = 2, pos = positions, fixed = fixed_nodes)
        #
        print(pos)

        nx.draw_networkx_nodes(G, pos,  node_size = 200, node_color = node_colors, alpha = node_alpha)

        # for node, (x, y) in pos.items():
        #     if G.nodes[node]['resultant'] == highest_preference:
        #         text(x, y,' '+ node + ' ', ha='left', va='bottom', fontsize = 11, fontname = "Georgia")

        for node in G.nodes():
            #if G.nodes[node]['resultant'] == highest_preference:
            labeldict[node] = node #+ ' ' + str(int(G.nodes[node]['in_arrows'])) + ' ' + str(int(G.nodes[node]['out_arrows']))

        nx.draw_networkx_labels(G, pos, labels = labeldict
        , horizontalalignment = 'center'
        ,verticalalignment = 'bottom',
        font_family = 'Georgia' )

        nx.draw_networkx_edges(G, pos , arrows=True, 
            arrowsize=10,
            edge_color=edge_colors,
            alpha = 0.2,
            #edge_cmap=plt.cm.Blues,
            width=2)


        #nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        #nx.draw_networkx_edges(G, pos,  arrows=False)

        ax = plt.gca()
        ax.set_axis_off()
        #ax.set_title(actor, font = 'Georgia', fontsize = 12)
        ax.axis('off')
        
        #plt.savefig(output_path, format = "PNG")
        plt.show(block = False)