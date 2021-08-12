import data_02_conflictSolvers as solvers
import data_01_conflictModel as model
import data_03_gmcrUtilities as gmcrUtils
import gmcr
import numpy as np
from utilities import *



def getConflict(payoffs, decisions, baseline, actors, baseline_name):
    
    # pick out only the decisions over which a final decision needs to be made
    temp = payoffs[payoffs.index.isin(decisions)]

    # append the baseline 
    #payoffs1 = add_baseline(temp, baseline_name)
    payoffs1 = temp.append(baseline, ignore_index = True)
    
    # reset index names
    payoffs1.index = list(temp.index) + [baseline_name]
    payoffs1.index.name = 'decisions'

    # drop duplicates if any
    payoffs1 = payoffs1.reset_index().drop_duplicates()
    payoffs1 = payoffs1.set_index('decisions')

    # get utilities 
    payoffs1 = normalize(payoffs1)
    payoffs1.index.rename('decisions', inplace = True) 

    # pick out only the actors on the table
    payoffs1 = payoffs1[actors]

    # create a conflict
    myConflict = gmcr.model.ConflictModel(payoffs1, baseline_name)

    myConflict.useManualPreferenceRanking = True

    if 'hydropower_revenue' in actors:
        hydropower_revenue = gmcr.model.DecisionMaker(myConflict, "hydropower_revenue")
        myConflict.decisionMakers.append(hydropower_revenue)

    if 'atomic_power_plant_discharge' in actors:
        atomic_power_plant_discharge = gmcr.model.DecisionMaker(myConflict, "atomic_power_plant_discharge")
        myConflict.decisionMakers.append(atomic_power_plant_discharge)

    if 'baltimore_discharge' in actors:
        baltimore_discharge = gmcr.model.DecisionMaker(myConflict, "baltimore_discharge")
        myConflict.decisionMakers.append(baltimore_discharge)

    if 'chester_discharge' in actors:
        chester_discharge = gmcr.model.DecisionMaker(myConflict, "chester_discharge")
        myConflict.decisionMakers.append(chester_discharge)

    if 'recreation' in actors:
        recreation = gmcr.model.DecisionMaker(myConflict, "recreation")
        myConflict.decisionMakers.append(recreation)

    if 'environment' in actors:
        environment = gmcr.model.DecisionMaker(myConflict, "environment")
        myConflict.decisionMakers.append(environment)

    if 'flood_risk' in actors:
        flood_risk = gmcr.model.DecisionMaker(myConflict, "flood_risk")
        myConflict.decisionMakers.append(flood_risk)

    # Add all the decisions the players must chose from except the baseline 

    nr_players = len(actors)
    for i in range(0, nr_players):
        for j in decisions:
            myConflict.decisionMakers[i].addOption(j)


    myConflict.recalculateFeasibleStates()

    return myConflict, payoffs1

def getConflictSolver(myConflict, payoffs):

    nr_players = payoffs.shape[1]

    # Create a temporary list mapping each state to the policy decision that it would result in. Each state is a combination of strategies an actor chooses
    temp = [(x,y) for x,y in zip([feas for feas in myConflict.feasibles.name], [feas for feas in myConflict.feasibles])]
    #print(temp)
    # create a dictionary mapping every policy to all the states where it gets picked
    dict = {}
    for i in set([feas for feas in myConflict.feasibles.name]):
        dict[i] = []
    for i in temp:
        dict[i[0]].append(i[1])

    # create ordinal ranking between all the options
    # rename the baseline decision
    ranks = payoffs.rank(ascending = False)
    ranks = ranks.reset_index()


    actors = [dm.name for dm in myConflict.decisionMakers]

    # create a preference ranking for each actor based on their ranking
    # Add that to the conflict
    for i in range((nr_players)):

        rank_order = list(zip(list(ranks['decisions']), list(ranks[actors[i]])))
        index_map = {j:v for j, v in (rank_order)}
        #print(index_map)
        sorted_list = sorted(dict.items(), key=lambda x: index_map[x[0]])
        myConflict.decisionMakers[i].preferenceRanking = list(list(zip(*sorted_list))[1])

    # calculate preferences for every actor based on the ranking
    for i in range((nr_players)):
        myConflict.decisionMakers[i].calculatePreferences()

    #%%time
    lSolver = solvers.LogicalSolver(myConflict)

    lSolver.findEquilibria()

    decision_names = np.array(myConflict.feasibles.name)


    stabilities = {'Nash Equilibrium': [set(decision_names[lSolver.nashEquilibria])],
                    'GMR Equilirbium': [set(decision_names[lSolver.gmrEquilibria])],
                    'SEQ Equilibrium': [set(decision_names[lSolver.seqEquilibria])],
                    'SIM Equilibrium': [set(decision_names[lSolver.simEquilibria])],
                    'SEQ SIM Equilibrium': [set(decision_names[lSolver.seqSimEquilibria])],
                    'SMR Equilirbium': [set(decision_names[lSolver.smrEquilibria])]}
    
    return myConflict, lSolver, stabilities