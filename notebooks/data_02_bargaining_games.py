import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt
import math
from utilities import *

class bargainingGames:
    '''Uses to create a bargaining game. 
    Specify payoffs, desired actors and desired aggregation method - resultant, cardinal_resultant, total_utility
    Steps - 
    1. Converts payofss to utilities
    2. Creates adjacency matrix 
    3. Plots each actors preference direction
    4. Calculates preference aggregation across all actors based on the desired method
    '''
    def __init__(self, payoffs, desired_actors, baseline):
        print(baseline)
        self.payoffs = add_baseline(payoffs, baseline)
        self.desired_actors = desired_actors

        fixed_positions = [[1.89163192, 0.83187892],
                                [1.74778059, -0.90970226],
                                [-1.22156597, 1.65536481],
                                [-1.30482134, -1.53638689],
                                [0.38242764, -2.0],
                                [0.50341964, 1.88134242],
                                [-1.99887247, 0.07750299],
                                [-0.00233494, -0.00270822]]
        self.positions = dict(zip(self.payoffs.index,fixed_positions))

        #print(self.positions)

    # def payoff_baseline(self, payoffs, baseline = None):
    #     df = payoffs.copy()
    #     if baseline == 'Zero Baseline':

    #         df = df.append({'hydropower_revenue':0.0, 'atomic_power_plant_discharge':0.0,
    #     'baltimore_discharge':0.00, 'chester_discharge':0.00, 'recreation':0.00, 'environment':-1,
    #     'flood_risk':-3}, ignore_index = True)

    #         df.index = list(payoffs.index) +  ['baseline']

    #     if baseline == 'Status Quo':

    #         df = df.append({'hydropower_revenue':39.30, 'atomic_power_plant_discharge':0.633,
    #     'baltimore_discharge':1.00, 'chester_discharge':1.00, 'recreation':0.00, 'environment':-0.1058,
    #     'flood_risk':0}, ignore_index = True)

    #         df.index = list(payoffs.index) +  ['baseline']

    #     if baseline == 'Status Quo 2':

    #         df = df.append({'hydropower_revenue':39.30, 'atomic_power_plant_discharge':0.633,
    #     'baltimore_discharge':1.00, 'chester_discharge':1.00, 'recreation':0.00, 'environment':-0.1058,
    #     'flood_risk':-1.633}, ignore_index = True)

    #         df.index = list(payoffs.index) +  ['baseline']
        
        #if baseline == 'No Baseline':

            
            #positions = dict(zip(df.index,fixed_positions))
            #positions = {'policy7': ([1.89163192, 0.83187892]), 'policy15': ([ 1.74778059, -0.90970226]), 'policy16': ([-1.22156597,  1.65536481]), 'policy28': ([-1.30482134, -1.53638689]), 'policy30': ([ 0.38242764, -2.        ]), 'policy46': ([0.50341964, 1.88134242])}

        # fixed_positions = [[1.89163192, 0.83187892],
        #                         [1.74778059, -0.90970226],
        #                         [-1.22156597, 1.65536481],
        #                         [-1.30482134, -1.53638689],
        #                         [0.38242764, -2.0],
        #                         [0.50341964, 1.88134242],
        #                         [-1.99887247, 0.07750299],
        #                         [-0.00233494, -0.00270822]]
        # positions = dict(zip(df.index,fixed_positions))

        # return df, positions

    def set_parameters(self, utility_function):

        decision_names = self.payoffs.T.columns

        self.normalized_payoffs = utility_function(self.payoffs).reset_index()
        #print(normalized_payoffs)


        self.decision_names = decision_names
        self.strategies = self.normalized_payoffs.T#.drop('index')
        self.strategies.columns = list(self.strategies.iloc[0])
        self.strategies.drop('decision', inplace=True)
    
        self.adj_matrix = pd.crosstab(self.strategies.T.index, self.strategies.T.index).reset_index()
        self.adj_matrix.index = list(self.adj_matrix['row_0'])
        self.adj_matrix.drop(columns = 'row_0', inplace = True)
        for col in self.adj_matrix.columns:
            self.adj_matrix[col].values[:] = 0
        
        self.fixed_nodes = self.positions.keys()

        


    def preference_direction(self, actor):

        # strategies is simply the dataframe of normalized payoffs for each actor
        strategies = self.strategies
        # adjacency matrix is a dataframe with policies as rows and policies as columns. It maps the direction in which one policy is preferred to another
        adj_matrix = self.adj_matrix
        # ranks is the ordinal value 
        ranks = strategies.rank(ascending = False, axis=1)

        
        #print(ranks)
     

        G = nx.DiGraph()

        for index, values in strategies.items():
            G.add_node(index)
            G.nodes[index]['total_utility'] = 0
            G.nodes[index]['least_preferred'] = 0
            for itemName, itemContent in values.items():
                for i in actor:
                    if itemName == i:
                        G.nodes[index][itemName] = itemContent
                        G.nodes[index]['total_utility'] += itemContent
                        #print(itemName, itemContent)
                        #print(np.min(self.normalized_payoffs[i]))
                        if itemContent == np.min(self.normalized_payoffs[i]):
                            G.nodes[index]['least_preferred'] +=1


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

        fig, axes = plt.subplots(rows,columns, figsize = (20,10))
        actors = actors

        if fixedPositions:
            positions = self.positions
            fixed_nodes = self.fixed_nodes

        if criteria == 'resultant':
            outcome_weight = 'weight'
        elif criteria == 'cardinal_resultant':
            outcome_weight = 'cardinal_weight'

        for i in range(len(graphs)):
            
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
                if criteria != 'total_utility':
                    if G.nodes[node]['least_preferred'] >0:
                            #print(G.nodes[node])
                            node_colors.append("darkred")
                    elif G.nodes[node][criteria] == highest_preference:

                    #node_colors.append(G.nodes[node][criteria])
                        node_colors.append("darkblue")
                    else:
                        node_colors.append("lightblue")

                # else:
                #     node_colors.append("lightblue")
        

            if fixedPositions:
                pos = nx.spring_layout(G, k = 0.5, weight = 0, scale = 2, pos = positions, fixed = fixed_nodes)
            else:
                pos = nx.spring_layout(G, k = 0.5, weight = 0, scale = 2)
        

            longest_path = nx.dag_longest_path(G)

            longest_edges = []
            edge_colors = []
            edge_alpha = []
        
            
    
            nx.draw_networkx_nodes(G, pos,  node_size = 200, cmap=plt.cm.Blues, ax = ax1, node_color = node_colors)#, alpha = node_alpha, ax = ax1)

            # for node, (x, y) in pos.items():
            # #     if G.nodes[node]['resultant'] == highest_preference:
            #     text(x, y,' '+ node + ' ', ha='left', va='bottom', fontsize = 11, fontname = "Georgia")

            for node in G.nodes():
                # if G.nodes[node]['resultant'] == highest_preference or G.nodes[node]['resultant'] == lowest_preference:
                labeldict[node] = node

            nx.draw_networkx_labels(G, pos, labels = labeldict, ax = ax1
            , horizontalalignment = 'center'
            ,verticalalignment = 'top',
            font_family = 'Helvetica' )

            for edge in G.edges():
                edge_colors.append(G.edges[edge][outcome_weight]*3)

            #edge_colors = np.array(edge_colors, dtype = "object")

            #edge_alpha = np.array(edge_alpha, dtype = "object")

            #print(edge_colors)

            nx.draw_networkx_edges(G, pos , arrows=True, 
                arrowsize=12,
                edge_color= "grey",
                alpha = 1,
                width= 1.2,
                ax = ax1)

            # for i, arc in enumerate(arcs):  # change alpha values of arcs
            #     arc.set_alpha(edge_alpha[i])

            #nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
            #nx.draw_networkx_edges(G, pos,  arrows=False)

            ax = plt.gca()
            ax.set_axis_off()
            ax1.set_title(actor, font = 'Georgia', fontsize = 14)
            ax1.axis('off')
            #ax.title(actor)
        
        #return fig

        plt.savefig('../visuals/06/fallback_1.png', format = "PNG")
        plt.tight_layout()
        plt.show(block = False)

    def combined_graph(self, all_outcomes, criteria, games, baselines,fixedPositions):
        #print(all_outcomes)

        if criteria == 'resultant':
            outcome_weight = 'weight'
        elif criteria == 'cardinal_resultant':
            outcome_weight = 'cardinal_weight'

        columns = len(baselines)
        rows = int(math.ceil((len(baselines))/columns))

        fig, axes = plt.subplots(rows,columns, figsize = (12,4))
        baselines = baselines
       
        for index, all_outcome in enumerate(all_outcomes):
            #print(all_outcome)
            G = all_outcome

            edge_colors = "grey"

            positions = games[index].positions
            fixed_nodes = games[index].fixed_nodes
            
            ax = axes.flatten()
            ax1 = ax[index]

            highest_preference = max([G.nodes[n][criteria] 
                                    for n in G.nodes() if G.nodes[n]['least_preferred']==0]) 
            labeldict = {}
            node_colors = []

            for node in G.nodes():
                if G.nodes[node]['least_preferred'] >0:

                    node_colors.append('darkred')
                else:
                    if G.nodes[node][criteria] == highest_preference:
                        node_colors.append('darkblue')
                    else:

                        node_colors.append('lightblue')


            if fixedPositions:
                pos = nx.spring_layout(G, k = 0.5, weight = 0, scale = 2, pos = positions, fixed = fixed_nodes)
            else:
                pos = nx.spring_layout(G, k = 0.5, weight = 0, scale = 2)


            nx.draw_networkx_nodes(G, pos,  node_size = 200, cmap=plt.cm.RdYlBu, node_color = node_colors, ax = ax1)

            # for node, (x, y) in pos.items():
            #     if G.nodes[node]['resultant'] == highest_preference:
            #         text(x, y,' '+ node + ' ', ha='left', va='bottom', fontsize = 11, fontname = "Georgia")

            for node in G.nodes():
                labeldict[node] = str(G.nodes[node][criteria]) #node + str(" ") +str(G.nodes[node][criteria]) #+ ' ' + str(int(G.nodes[node]['in_arrows'])) + ' ' + str(int(G.nodes[node]['out_arrows']))

            nx.draw_networkx_labels(G, pos, labels = labeldict
            , horizontalalignment = 'right'
            ,verticalalignment = 'top',
            font_weight= 15,
            font_family = 'Helvetica', ax= ax1 )


            edge_colors = []

            for edge in G.edges():
                edge_colors.append(G.edges[edge][outcome_weight])

            arcs = nx.draw_networkx_edges(G, pos , arrows=True, 
                arrowsize=12,
                edge_color= "lightgrey",
                alpha = 0.8,
                edge_cmap=plt.cm.Blues,
                width=1.2,
                ax = ax1)

            #print(edge_alpha)

            # for i, arc in enumerate(arcs):  # change alpha values of arcs
            #     arc.set_alpha(edge_alpha[i])

            #nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
            #nx.draw_networkx_edges(G, pos,  arrows=False)

            ax = plt.gca()
            ax.set_axis_off()
            ax1.set_title(baselines[index], font = 'Georgia', fontsize = 12)
            ax1.axis('off')
            
        plt.savefig('../visuals/06/fallback_2.png', format = "PNG")
        plt.tight_layout()
        plt.show(block = False)

        #return pos