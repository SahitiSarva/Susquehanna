import pandas as pd 
from sklearn import preprocessing
import numpy as np
import ast
import networkx as nx

#class noncooperativegames:
  #  def __init__(self, player1, player2):
   #     self.player1 = player1
    #    self.player2 = player2



def votes(self, normalized_payoffs):
    player1 = self.player1
    player2 = self.player2 

    vote_space_p1 = np.zeros((7,7))
    vote_space_p2 = np.zeros((7,7))
    vote_outcomes = {}
    count = 0
    for i in range(0,7):
        for j in range(0,7):
            if i == j:
                vote_space_p1[i][j] = normalized_payoffs[player1][i]
                vote_space_p2[i][j] = normalized_payoffs[player2][i]
                vote_outcomes[count] = {'decision':normalized_payoffs['decision'][i], 'vote': (i,j)}
            else:
                total_utility1 = normalized_payoffs[player1][i] + normalized_payoffs[player2][i]
                total_utility2 = normalized_payoffs[player1][j] + normalized_payoffs[player2][j]
                if total_utility1 > total_utility2:
                    vote_space_p1[i][j] = normalized_payoffs[player1][i]
                    vote_space_p2[i][j] = normalized_payoffs[player2][i]
                    vote_outcomes[count] = {'decision':normalized_payoffs['decision'][i], 'vote': (i,j)}

                elif total_utility1 < total_utility2:
                    vote_space_p1[i][j] = normalized_payoffs[player1][j]
                    vote_space_p2[i][j] = normalized_payoffs[player2][j]
                    vote_outcomes[count] = {'decision':normalized_payoffs['decision'][j], 'vote': (i,j)}
            count +=1

    return vote_space_p1, vote_space_p2, vote_outcomes


def graphs(vote_space_p1, vote_space_p2):

    nr_decisions = vote_space_p1.shape[0]
    print(nr_decisions, "decisions")

    G = nx.DiGraph()

    node = 0
    for i in range(0,nr_decisions):
        for j in range(0,nr_decisions):
            G.add_node(node)
            G.nodes[node]['v1'] = i
            G.nodes[node]['v2'] = j
            G.nodes[node]['u1'] = vote_space_p1[i][j]
            G.nodes[node]['u2'] = vote_space_p2[i][j]
            node += 1

    for n,v in G.nodes(data = True):
        p2_vote = G.nodes[n]['v2']
        p1_vote = G.nodes[n]['v1']
        p1_moves = [n for n,v in G.nodes(data = True) if v['v2'] == p2_vote]
        p2_moves = [n for n,v in G.nodes(data = True) if v['v1'] == p1_vote]
        #print(my_moves)
        for node in p1_moves:
            if G.nodes[node]['u1'] > G.nodes[n]['u1']:
                G.add_edge(n, node, weight = 1, cardinal_weight = G.nodes[node]['u1'] - G.nodes[n]['u1'], player = 1, edge_type = 'useful')
                #print(n, node, "player1")
            else:
                G.add_edge(n, node, weight = 0, player = 1, edge_type = 'possible')
        for node in p2_moves:
            #print("entered loop")
            if G.nodes[node]['u2'] > G.nodes[n]['u2']:
                G.add_edge(n, node, weight = 1, cardinal_weight = G.nodes[node]['u2'] - G.nodes[n]['u2'], player = 2, edge_type = 'useful')
                #print(n, node, "player2")
            else:
                G.add_edge(n, node, weight = 0, player = 2, edge_type = 'possible')

    for node in G.nodes():
        node_in_weight = 0
        node_out_weight = 0
        for i in G.in_edges(node):
            node_in_weight += G.edges[i]['weight']
            #node_in_cardinal_weight += G.edges[i]['cardinal_weight']
        for i in G.out_edges(node):
            node_out_weight += G.edges[i]['weight']
            #node_out_cardinal_weight += G.edges[i]['cardinal_weight']
        G.nodes[node]['resultant'] = node_in_weight - node_out_weight
        #G.nodes[node]['cardinal_resultant'] = node_in_cardinal_weight - node_out_cardinal_weight
        G.nodes[node]['in_arrows'] = node_in_weight
        G.nodes[node]['out_arrows'] = node_out_weight
            
    return G

def gmr(G, player, utility):
    equilibria = []
    for n in G.nodes:
        intermediate_nodes = [x[1] for x in list(nx.dfs_edges(G, source = n, depth_limit = 1)) if G.edges[x]['player'] == player and G.edges[x]['edge_type']=='useful']
        #print(intermediate_nodes)
        for i in intermediate_nodes:
            edges_list = list(nx.dfs_edges(G, source = i, depth_limit = 1))
            tertiary_nodes = [G.nodes[edge[1]][utility] for edge in edges_list if G.edges[edge]['player'] != player]
            #print(tertiary_nodes)
            if all(tertiary_nodes <= G.nodes[n][utility]):
                equilibria.append(n)
    equilibria = list(dict.fromkeys(equilibria))

    return equilibria

def payoff_matrix(player1, player2, baseline ,normalized_payoffs):
        # player1 = self.player1
        # player2 = self.player2
        # 
        nr_decisions = normalized_payoffs.shape[0] 

        payoff_matrix_1 = np.zeros((nr_decisions,nr_decisions))
        payoff_matrix_2 = np.zeros((nr_decisions,nr_decisions))
        game_outcomes = {}
        count = 0
        for i in range(0,nr_decisions):
            for j in range(0,nr_decisions):
                if i == j:
                    payoff_matrix_1[i][j] = normalized_payoffs[player1][i]
                    payoff_matrix_2[i][j] = normalized_payoffs[player2][i]
                    game_outcomes[count] = {'decision':normalized_payoffs['decision'][i]}
                else:
                    payoff_matrix_1[i][j] = baseline[player1]#normalized_payoffs[player1][7]
                    payoff_matrix_2[i][j] = baseline[player2]#normalized_payoffs[player2][7]
                    game_outcomes[count] = {'decision':'baseline'}

                count +=1

        return payoff_matrix_1, payoff_matrix_2, game_outcomes