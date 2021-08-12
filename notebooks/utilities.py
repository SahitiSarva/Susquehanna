import pandas as pd 
from sklearn import preprocessing
import numpy as np
import ast
import networkx as nx
import pickle
import math

""""
********************* Data transformations ************************
- load_data
*****************************************************************
"""


def load_data(file_name, method = ''):
    """"Loads Data from a dictionary and names all the policies sequentially"""
    

    with open(file_name, "rb") as f:
        payoffs = pickle.load(f)
        payoffs = pd.DataFrame(payoffs)

    payoffs = payoffs.T

    payoffs = payoffs.drop(columns = ["coalition_appchester", 'coalition_discharge','coalition_watersupply', 'flood_duration'], errors = 'ignore')


    payoffs = payoffs*-1
    payoffs = payoffs.drop_duplicates()
    payoffs.index = [f"policy{i+1}" for i in range(payoffs.shape[0])]
    payoffs.index.rename('decision', inplace = True)
    
    not_optimal = ['policy52',
 'policy104',
 'policy107',
 'policy55',
 'policy9',
 'policy16',
 'policy25',
 'policy29',
 'policy33',
 'policy44',
 'policy51',
 'policy58',
 'policy90',
 'policy99',
 'policy34',
 'policy31',
 'policy21',
 'policy88',
 'policy71']

    #payoffs = payoffs[~(payoffs.index.isin(not_optimal))]


    return payoffs



def format_payoffs(actor, payoffs):
    if actor in ['atomic_power_plant_discharge',
       'baltimore_discharge', 'chester_discharge', 'recreation', 'envrionment']:
       return str(str(round(payoffs*100,2))+" %")
    elif actor == 'flood_risk':
        return str(round(abs(payoffs),2)) + " m"
    elif actor == 'environment':
        return str(str(abs(round(payoffs*100,2)))+" %")
    elif actor == 'hydropower_revenue':
        return str(round(payoffs)) + " Mn.$"
    else:
        return "something is wrong"

""""
********************* Utility Functions ************************
- normalize     : Linear transformation with min max scaler
- standardize   : Linear transformation with mean as 0 and std as 1
- prioritarian  : Non discounted prioritarian SWF
*****************************************************************
"""


def normalize(payoffs, rho=None, withNormalization = None):
    "Normalizes data - primary linear utility function. Uses a min-max scaler"

    decision_names = payoffs.T.columns
    min_max_scaler = preprocessing.MinMaxScaler() 
    x_scaled = min_max_scaler.fit_transform(payoffs)
    #x_scaled
    normalized_payoffs = pd.DataFrame(x_scaled).reset_index(drop = True)
    normalized_payoffs.columns = list(payoffs.columns)
    normalized_payoffs.index = decision_names 
    normalized_payoffs.reset_index(inplace=True)
    normalized_payoffs.columns = ['decision'] + list(payoffs.columns)
    normalized_payoffs.reset_index(inplace=True)

    normalized_payoffs = normalized_payoffs.set_index('index').reset_index(drop = True)
    normalized_payoffs = normalized_payoffs.set_index('decision')
    #normalized_payoffs = normalized_payoffs + 1




    return normalized_payoffs

def no_utility(payoffs, rho = None):
    return payoffs

def standardize(data):
    "Standardizes data with mean value as 0 and standard deviation as 1"


    refSet = np.array(data)
    nobjs = np.shape(refSet)[1]
    normObjs = np.zeros([np.shape(refSet)[0],nobjs])
    # for every policy
    for i in range(np.shape(refSet)[0]):
        # for every actor
        for j in range(nobjs):
            # take the square root of the deficit so it's less skewed
            if refSet[:,j].any()<0:
                refSet[:,j] = refSet[:,j]*-1
                normObjs[i,j] = (np.sqrt(refSet[i,j])-np.mean(np.sqrt(refSet[:,j])))/np.std(np.sqrt(refSet[:,j]))
            else:
                 normObjs[i,j] = (refSet[i,j]-np.mean(refSet[:,j]))/np.std(refSet[:,j])

    normObjs = pd.DataFrame(normObjs)
    normObjs.index = data.index
    normObjs.columns = data.columns

    return normObjs

def find_worst_for_all(refSet):#deficitIndex):
    nobjs = np.shape(refSet)[1]
    normObjs = np.array(refSet)
    # array of all policies
    dists = np.zeros(np.shape(refSet)[0])
    for i in range(len(dists)):
        for j in range(nobjs):

            # distance of that objective for that policy from the minimum
            # minimizing distance from the 
            dists[i] = dists[i] + (normObjs[i,j]-np.min(normObjs[:,j]))**2
            
    compromise = np.argmax(dists)

    
    return refSet.iloc[compromise].name

def get_marginal_utility_of_df(payoffs, rho=None, withNormalization = False):

    if rho==None:
        rho = 1.2
    df = payoffs.copy()
    

    for i in df.columns:
        if i in ['environment', 'flood_risk']:
            df[i] = df[i].apply(lambda x: get_marginal_utility(x, rho))
        else:
            df[i] = df[i].apply(lambda x: get_marginal_utility(x, rho))
    
    df.index.rename('decision', inplace = True)

    if withNormalization == True:
        df = normalize(df)

    return df

def get_marginal_utility(payoff=0, rho=None):
    """rho: risk aversion parameter"""

    if rho == None:
        rho = 1.2

    marginal_utility = 1-rho

    if payoff <0 and marginal_utility != 0:
    #     #payoff = payoff*-
         utility = -((math.pow(-payoff, marginal_utility)/marginal_utility )+20)
    
    if payoff >0 and marginal_utility !=0:
        utility = math.pow(payoff, marginal_utility)/marginal_utility + 20

    elif payoff >0 and marginal_utility == 0:
        utility = math.log(payoff) + 20

    #print(payoff)
    elif payoff <0 and marginal_utility == 0:
         utility = -(math.log(-payoff) + 20)
    #elif payoff == 0:
    #     utility = 0

    return utility 



def marginal_disutility(payoff=0, rho=None):

    if payoff < 0:
        payoff = payoff *-1

    if rho == None:
        rho = 1

    if rho >0:
        utility = 1 - math.exp(-1*rho*payoff)

    return utility




""""
********************* Fairness functions ************************
- get_utility_gini
- get_fairness_metrics
- find_egalitarian_outcome
- find_gini_index
- find_least_envy
- find_envy

TODO: Why are there two functions?
TODO: Every function needs to do only one thing. Dont load data directly in functions
*****************************************************************
"""

def find_egalitarian_outcome(refSet):
    "Gets the outcome with the lowest gini index"

    gini_index = find_gini_index(refSet)

    return gini_index.iloc[np.argmin(gini_index)].name

def get_most_suffecientarian_outcome(payoffs, utility_function, gamma=None, rho=None):

    welfare = get_suffecienteerism_welfare(payoffs, utility_function)
    policy = welfare.iloc[np.argmax(welfare['suffecientarian_welfare'])]

    return policy.name

def get_most_prioritarian_outcome(payoffs, utility_function, rho = None):
    "Gets the most prioritarian outcome"

    welfare = get_prioritarian_swf_for_df(payoffs, utility_function, rho)

    return welfare.iloc[np.argmax(welfare)].name

def find_gini_index(refSet):
    "Finds the gini index of each policy in a dataframe. Formula from Cuillo et. al. (2020)"
    refSet = normalize(refSet)
    mean_x = np.mean(refSet, axis=1)
    denominator = 2*(mean_x)*np.power(refSet.shape[1],2)
    
    numerator = 0

    for i in refSet:
        numerator+= refSet.sub(list(refSet[i]), axis = 'rows').abs().sum(axis=1)

    gini_index = pd.DataFrame(numerator/denominator)

    return gini_index

def find_least_envy(refSet):
    "Not in use currently. Finds the envy of a particular df"

    envy = find_envy(refSet)

    return envy.iloc[np.argmin(envy)].name

def find_envy(refSet):

    #mean_x = np.mean(refSet, axis=1)
    #denominator = 2*np.power(mean_x,2)*refSet.shape[0]
    
    numerator = 0

    for i in refSet:
        numerator+= refSet.sub(list(refSet[i]), axis = 'rows').abs().sum(axis=1)

    envy = pd.DataFrame(numerator)

    return envy


def find_least_diff_from_mean(refSet):

    diff_from_mean = find_diff_from_mean(refSet)

    return diff_from_mean.iloc[np.argmin(diff_from_mean)].name

def find_diff_from_mean(refSet):

    mean_x = pd.DataFrame(np.mean(refSet, axis=1))
    #denominator = 2*np.power(mean_x,2)*refSet.shape[0]
    
    numerator = 0

    for i in refSet:
        numerator+= mean_x.sub(list(refSet[i]), axis = 'rows').abs().sum(axis=1)

    diff_from_mean = pd.DataFrame(numerator)

    return diff_from_mean

def get_all_fairness_metrics(payoffs, utility_function, gamma=None, rho = None):
    #print(gamma)
    utilities = utility_function(payoffs, rho)
    df = get_utility_gini(utilities)
    df1 = get_prioritarian_swf_for_df(payoffs, utility_function, gamma, rho)

    df2 = get_suffecienteerism_welfare(payoffs, utility_function, rho)
    df2 = df2[['suffecientarian_welfare']]
    df1.columns = ['prioritarian_welfare']
    df = df.merge(df1, on = 'decision')
    df = df.merge(df2, on = 'decision')

    df = df.set_index('decision')

    return df



def get_utility_gini(payoffs):
    utility = pd.DataFrame(payoffs.sum(axis=1))
    utility.columns = ['utility']
    #utility = (utility)

    equality = find_gini_index(payoffs)
    equality.columns = ['gini_index']
    #equality['gini_index'] = equality['gini_index']*-1
    #equality = (equality)

    scatter_df = equality.merge(utility, on = 'decision').reset_index()

    scatter_df = scatter_df.merge(payoffs, on = 'decision')

    return scatter_df

def get_fairness_metrics_with_baseline(baseline, utility_function, df):
    "Returns a datframe with gini index and utility values for each policy"

    #notable_policies = {'status_quo':'baseline', 'zero_baseline': 'baseline', 'baseline':'baseline'}
    notable_policies = {'baseline':'status quo', 
    'policy85': 'best from fallback', 
    'policy60':'competetive stability', 'policy67':'competetive stability',
    'policy75':'worst from fallback'}
    #{'policy2': 'chester', 'policy30': 'baltimore', 'policy37':'hydropower', 'policy64': 'atomicpower', 'baseline':'baseline'}


    #df = load_data("../model/data/processed/50yr_all2.pickle")
    df = add_baseline(df, baseline)    

    utilities = utility_function(df)

    scatter_df = get_utility_gini(utilities)

    scatter_df['policies'] = scatter_df['decision'].map(notable_policies)
    scatter_df = scatter_df.fillna('')


    return scatter_df

def get_fair_policies(utility_function, payoffs, rho=None):
    "Gets fair policies and returns a dictionary with ethical principle and policy name"
    selected_policies = {}

    df = utility_function(payoffs, rho)

    # utilitarian policy
    selected_policies.update({'utilitarian': df.iloc[np.argmax(df.sum(axis=1))].name})

    # prioritarian for each actor
    for actor in list(payoffs.columns):
        selected_policies.update({actor : list(df[df[actor]==max((df[actor]))].index)[0]})

    # egalitarian by gini index
    selected_policies.update({'egalitarian': find_egalitarian_outcome(df)})

    selected_policies.update({'prioritarian': get_most_prioritarian_outcome(payoffs, utility_function, rho = rho)})

    selected_policies.update({'suffecientarianism':get_most_suffecientarian_outcome(payoffs, utility_function)})

    # # egalitarian by envy
    # selected_policies.update({'egalitarian_envy': find_least_envy(df)})

    # #egalitarian by difference from the mean
    # selected_policies.update({'egalitarian_diff_from_mean' : find_least_diff_from_mean(df)})

    return selected_policies
""""
********************* Prioritarian functions ************************
- get_welfare
- get_prioritarian_welfare
- get_prioritarian_welfare_without_normalization
- get_prioritarian_swf_for_df
*****************************************************************
"""

def get_welfare(payoff, gamma=None, czero=None, rho =None):
    """gamma: social welfare transformation function
        czero: the zero bundle or the minimum utility value. This is an ethical decision
        utility: tranformed utility based on diminishing returns that is a positive value"""
  
    utility = get_marginal_utility(payoff, rho)

    if gamma == None:
        gamma = 1
    
    if rho ==None:
        rho =1.2

    marginal_utility = 1-gamma
    if czero != None:
        uzero = get_marginal_utility(czero, rho)
    else:
        uzero = 0
    
    if utility - uzero == 0:
            welfare = 0

    elif marginal_utility !=0:
        if utility - uzero <0 :
            #print("utility less than zero", utility, uzero)
            welfare = 0
            #welfare = -math.pow(-(utility-uzero), marginal_utility)/(1-gamma)
        else:
            # if utility - uzero == 0:
            #     print("utility almost zero")
            #     welfare = math.pow(0.0000000001, marginal_utility)
            welfare = math.pow((utility - uzero), marginal_utility)/(1-gamma)

    elif marginal_utility == 0:
        if utility - uzero < 0:
            welfare = 0
            #welfare = -math.log(-(utility-uzero))
        else:
            welfare = math.log(utility-uzero)
    
    # if welfare <0:
    #     welfare = welfare*-1

    return welfare

def get_welfare_of_df(utilities, gamma = None, rho=None):

    if rho == None:
        rho = 1.2

    if gamma==None:
        gamma = 1
    df = utilities.copy()
    
    for i in df.columns:
        df[i] = df[i].apply(lambda x: get_welfare(x, gamma))
    
    df.index.rename('decision', inplace = True)

    return df

def get_prioritarian_welfare(payoffs, utility_function, gamma=None, rho=None):

    actors = ['hydropower_revenue', 'atomic_power_plant_discharge',
       'baltimore_discharge', 'chester_discharge', 'environment',
       'flood_risk', 'recreation']


    utilities = utility_function(payoffs, rho)
    welfare = payoffs.copy()

    for i in utilities:
    
        if i == 'environment':
            #print(i, x)
            welfare[i] = (pd.DataFrame(welfare[i].apply(lambda x: get_welfare(x,gamma, -1, rho))))
        elif i in ['baltimore_discharge']:
            welfare[i] = (pd.DataFrame(welfare[i].apply(lambda x: get_welfare(x,gamma, 0.001, rho))))
        elif i in ['chester_discharge']:
            welfare[i] = (pd.DataFrame(welfare[i].apply(lambda x: get_welfare(x,gamma, 0.001, rho))))
        elif i in ['atomic_power_plant_discharge']:
            welfare[i] = (pd.DataFrame(welfare[i].apply(lambda x: get_welfare(x,gamma, 0.001, rho))))
        elif i in ['hydropower_revenue']:
            welfare[i] = (pd.DataFrame(welfare[i].apply(lambda x: get_welfare(x,gamma, 0.001, rho))))
        elif i == 'flood_risk':
            welfare[i] = (pd.DataFrame(welfare[i].apply(lambda x: get_welfare(x,gamma, -100, rho))))
        else:
            #print(i)
            welfare[i] = (pd.DataFrame(welfare[i].apply(lambda x: get_welfare(x,gamma, 0.5, rho))))

    welfare['utility'] = utilities[actors].sum(axis=1)

    welfare['prioritarian_welfare'] = welfare[actors].sum(axis =1)


    return welfare

def get_suffecienteerism_welfare(payoffs, utility_function, gamma=None, rho=None):

    if gamma == None:
        gamma = 0.99

    actors = ['hydropower_revenue', 'atomic_power_plant_discharge',
       'baltimore_discharge', 'chester_discharge', 'environment',
       'flood_risk', 'recreation']

    utilities = utility_function(payoffs, rho)
    welfare = payoffs.copy()

    for i in utilities:
        #print(i)
        if i == 'environment':
            welfare[i] = (pd.DataFrame(welfare[i].apply(lambda x: get_welfare(x,gamma,- 0.106, rho))))
        elif i in ['baltimore_discharge', 'chester_discharge']:
            welfare[i] = (pd.DataFrame(welfare[i].apply(lambda x: get_welfare(x,gamma, 0.3, rho))))
        elif i in ['atomic_power_plant_discharge']:
            welfare[i] = (pd.DataFrame(welfare[i].apply(lambda x: get_welfare(x,gamma, 0.6, rho))))
        elif i in ['hydropower_revenue']:
            welfare[i] = (pd.DataFrame(welfare[i].apply(lambda x: get_welfare(x,gamma, 30, rho))))
        elif i == 'flood_risk':
            welfare[i] = (pd.DataFrame(welfare[i].apply(lambda x: get_welfare(x,gamma, -0.001, rho))))
        else:
            #print(i)
            welfare[i] = (pd.DataFrame(welfare[i].apply(lambda x: get_welfare(x,gamma, 0.01, rho))))

    welfare['utility'] = utilities[actors].sum(axis=1)

    welfare['suffecientarian_welfare'] = welfare[actors].sum(axis =1)


    return welfare



def get_prioritarian_swf_for_df(payoffs, utility_function, gamma=None, rho=None):


    welfare = pd.DataFrame(get_prioritarian_welfare(payoffs, utility_function, gamma, rho)['prioritarian_welfare'])
    welfare.columns = ['prioritarian_welfare']

    return welfare

def get_fairness_tradeoffs_df(payoffs):

    df1 = pd.DataFrame(columns = ['gini_index', 'utility', 'hydropower_revenue',
        'atomic_power_plant_discharge', 'baltimore_discharge',
        'chester_discharge', 'recreation', 'environment', 'flood_risk',
        'prioritarian_welfare', 'suffecientarian_welfare', 'rho', 'gamma'])

    for gamma in np.arange(0,4,1).tolist():
        for rho in np.arange(0,4,1).tolist() + [1.2]:
            #print(rho)
            df = (get_all_fairness_metrics(payoffs, get_marginal_utility_of_df, gamma =gamma , rho =rho))
            df['gini_index'] = df['gini_index']*-1
            df = normalize(df)
            #df2 = normalize(df[['utility', 'prioritarian_welfare']])
            #df = df[['gini_index']]
            #df = df.merge(df2, on = 'decision')
            #df['utility'] = df1['utility']
            #df['prioritarian_welfare'] = df1['prioritarian_welfare']
            df['rho'] = rho
            df['gamma'] = gamma
            df1 = df1.append(df)


    return df1


""""
********************* Stability functions ************************
- get_fallback_df: Creates a dataframe with fair policies and ranking from fallback bargaining
Non cooperative stability 
- get_formatted_decisions: Formats decisions into tuples of two or three each depending on the length of decisions
- get_stable_policies: Run the graph model for conflict resolution and get stable solutions
*****************************************************************
"""
from gmcr_susquehanna import getConflictSolver, getConflict

def get_formatted_decisions(stable_decisions, nr_decisions = 2):
    decisions = []
    for i in range(0, len(stable_decisions), nr_decisions):
        if i+nr_decisions+1 == len(stable_decisions):
            decisions.append(stable_decisions[i:i+nr_decisions+1])
            break
        else:
            decisions.append(stable_decisions[i:i+nr_decisions])
    return decisions

def get_stable_policies(stability_analysis, baselines, decisions,payoffs, actors):

    run_num = len(stability_analysis)
    for key, value in baselines.items():
        for decision in decisions:  
            stability_analysis[run_num] = {'baseline':key, 'policies':decision, 'stabilities':{}, 'actors':actors}

            myConflict, payoffs1 = getConflict(payoffs=payoffs, baseline=value, decisions=decision, actors = actors, baseline_name = key)
            myConflict, lSolver, stability_analysis[run_num]['stabilities'] = getConflictSolver(myConflict, payoffs1)
            run_num += 1

    stability = get_stability_df(stability_analysis)[['baseline', 'policies', 'stability', 'index', 'actors']]
    stable_decisions = list(set(stability[ (stability['stability']>0) & (stability['baseline']!=stability['policies'])]['policies']))

    return stability_analysis, stability, stable_decisions

def get_fallback_df(all_outcomes, baselines, utility_function, payoffs, actors, fair_policies):
        
    fallback = pd.DataFrame(columns = ['decision', 'resultant', 'least_preferred', 'baseline'])

    for idx, outcome in enumerate(all_outcomes):
        fallback_temp = [(node, outcome.nodes[node]['resultant'],outcome.nodes[node]['least_preferred'])  for node in outcome.nodes]
        fallback_temp = pd.DataFrame(fallback_temp)
        fallback_temp.columns = ['decision', 'resultant', 'least_preferred']
        fallback_temp['baseline'] = baselines[idx]
        fallback = fallback.append(fallback_temp)

    #fallback = fallback[fallback['least_preferred']==0]
    fallback['policies'] = fallback['decision'].map(fair_policies)
    fallback['policies'] = fallback['policies'].fillna('other')

    fairness = pd.DataFrame(columns =['decision', 'gini_index', 'utility', 'policies', 'baseline'] + actors )
    for baseline in baselines:
        temp = get_fairness_metrics_with_baseline(baseline, utility_function, payoffs)
        temp['baseline'] = baseline
        fairness = fairness.append(temp)

    df = fairness.merge(fallback, on = ['decision', 'baseline'])

    df['color'] = df.apply(lambda x: determine_color(x['resultant'], x['least_preferred'], df), axis=1)

    return df

def determine_color(value, least_preferred, df):
    temp = df[(df['baseline']=='No Baseline')].sort_values(by = 'resultant', ascending = False)
    resultant = temp['resultant']
    if least_preferred > 0:
        return 'least preferred by atleast one actor'
    elif value > np.percentile(resultant, 95):
        return 'highly preferred (in game without disagreement point)'
    elif value <= 0: #np.percentile(resultant, 15) : 
        return 'unpreferred (in game without disagreement point)'
    elif  value > 0: #np.percentile(resultant, 50):
        return 'moderately preferred (in game without disagreement point)'



def add_baseline(payoffs, baseline):
    "Adds baseline data to an existing payoff df"

    df = payoffs.copy()

    if baseline == 'Zero Baseline':

        df = df.append({'hydropower_revenue':0.000000001, 'atomic_power_plant_discharge':0.000000001,
    'baltimore_discharge':0.000000001, 'chester_discharge':0.000000001, 'recreation':0.000000001, 'environment':-1,
    'flood_risk':-5}, ignore_index = True)

        df.index = list(payoffs.index) +  ['baseline']

    if baseline == 'Status Quo':

        df = df.append({'hydropower_revenue':39.30, 'atomic_power_plant_discharge':0.633,
    'baltimore_discharge':1.00, 'chester_discharge':1.00, 'recreation':0.001, 'environment':-0.1058,
    'flood_risk':-0.00098}, ignore_index = True)

        df.index = list(payoffs.index) +  ['baseline']

    if baseline == 'Status Quo 2':

        df = df.append({'hydropower_revenue':50.0, 'atomic_power_plant_discharge':1,
    'baltimore_discharge':1.00, 'chester_discharge':1.00, 'recreation':0.000000001, 'environment':-0.1058,
    'flood_risk':0}, ignore_index = True)

        df.index = list(payoffs.index) +  ['baseline']
    
    if baseline == 'No Baseline':
        
        df.index = list(payoffs.index)
    
    return df

def add_multiple_baselines(payoffs, baseline):
    "Adds baseline data to an existing payoff df"

    df = payoffs.copy()

    if baseline == 'Zero Baseline':

        df = df.append({'hydropower_revenue':0.000000001, 'atomic_power_plant_discharge':0.000000001,
    'baltimore_discharge':0.000000001, 'chester_discharge':0.000000001, 'recreation':0.000000001, 'environment':-1,
    'flood_risk':-5}, ignore_index = True)

        df.index = list(payoffs.index) +  ['Get nothing']

    if baseline == 'Status Quo':

        df = df.append({'hydropower_revenue':39.30, 'atomic_power_plant_discharge':0.633,
    'baltimore_discharge':1.00, 'chester_discharge':1.00, 'recreation':0.001, 'environment':-0.1058,
    'flood_risk':-0.00098}, ignore_index = True)

        df.index = list(payoffs.index) +  ['Susqstatusquo']

    if baseline == 'Status Quo 2':

        df = df.append({'hydropower_revenue':50.0, 'atomic_power_plant_discharge':1,
    'baltimore_discharge':1.00, 'chester_discharge':1.00, 'recreation':0.000000001, 'environment':-0.1058,
    'flood_risk':0}, ignore_index = True)

        df.index = list(payoffs.index) +  ['baseline']
    
    if baseline == 'No Baseline':
        
        df.index = list(payoffs.index)
    
    return df




def get_equilibrium_count(eq, policy):
    if policy in set(eq):
        return 1
    else:
        return 0

def interim_func(x):
    if isinstance(x, list):
        return list(x[0])
    else:
        return ['None']
        

def get_stability_df(stability_analysis):
    "Used in non cooperative game theory notebook to translate the stability dictionary to a readable format"


    stability_dict = stability_analysis.copy()


    df = pd.DataFrame(stability_dict).T.reset_index()


    df1 = df['stabilities'].apply(pd.Series)


    for i in df1:
        df1[i] = df1[i].apply(lambda x: interim_func(x))

    df = df.reset_index()
    df1 = df1.reset_index()
    
    df3 = df.merge(df1)
    df3['policies'] = df3.apply(lambda x: x['policies']+ [x['baseline']], axis=1)
    df3 = df3.explode('policies')

    for i in ['Nash Equilibrium', 'GMR Equilirbium', 'SEQ Equilibrium',
        'SIM Equilibrium', 'SEQ SIM Equilibrium', 'SMR Equilirbium']:
        df3[i] = df3.apply(lambda x: get_equilibrium_count(x[i], x['policies']), axis=1 )

    df3 = df3.drop(columns = ['stabilities'], errors = 'ignore')
    df3 = df3.groupby(['baseline', 'policies', 'index']).sum().reset_index()

    df3['stability'] = df3[['Nash Equilibrium', 'GMR Equilirbium', 'SEQ Equilibrium',
        'SIM Equilibrium', 'SEQ SIM Equilibrium', 'SMR Equilirbium']].sum(axis=1)
    
    df3 = df3.sort_values(by='index')
    df3['actors'] = df3.merge(df, on = 'index')['actors']

    return df3




def get_stable_df_from_dict(stability_analysis):

    stability = get_stability_df(stability_analysis)[['baseline', 'policies', 'stability', 'index', 'actors']]
    stability = stability[stability['stability']>0][['policies', 'stability', 'baseline', 'actors']]
    #stable_decisions = list(set(stability[ (stability['stability']>0)][['policies', 'stability']]))

    return stability


import matplotlib.cm as cm

def get_percentile(range, x, p, returnColor=False):
    if returnColor == False:
            
        if x > np.percentile(range, p):
            return 1
        elif x > np.percentile(range, 100-p):
            return 2
        else:
            return 3
    else:
        if x > np.percentile(range, p):
            
            return 'yellow'#cm.viridis(0)
        elif x > np.percentile(range, 100-p):
            
            return '#1F968B'
        else:
            
            return '#440154'

def get_percentage(p, x, returnColor = False):
    if returnColor == False:
            
        if x >= p/100:
            return 1
        elif x >= (100-p)/100:
            return 2
        else:
            return 3
    else:
        if x >= p/100:
            
            return 'yellow'#cm.viridis(0)
        elif x >= (100-p)/100:
            
            return '#1F968B'
        else:
            
            return '#440154'

def get_color(gini, utility, priority):
    if gini == utility and utility == priority and utility == 1:
        return 0
    # elif gini == 1:
    #     return 0.5
    # elif utility==1:
    #     return 0
    # elif priority ==1:
    #     return 0.25
    else:
        return 1
