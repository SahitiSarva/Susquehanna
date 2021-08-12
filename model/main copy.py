from susquehanna_model import susquehannaModel

import pandas as pd
import glob
import pickle


inputs = 4 # inputs
outputs = 4 # outputs - just the downstream discharge outputs. 
RBFs = inputs+2 # RBFs
n_years = 1

# Initialize the object
susquehanna_model = susquehannaModel(108.5, 505.0, 5, n_years)

susquehanna_model.load_data()

susquehanna_model.setRBF(RBFs, inputs, outputs)

# Running experiments

# Code to run all the decisions in the folder
#for name in glob.glob('./decisions/decisions_*[0-9]_.txt'):


outcomes = {}
for name in glob.glob('./best_decisions/best_app*'):
    print(name)

    decision = name.split("\\")[1].split('.')[0]
    df = pd.read_csv(name)
    levers = df.set_index('description').to_dict()['variable']

    outcomes[decision] = susquehanna_model.evaluateMC(levers)


output_filename = './output/processed/test.pickle'
#outcomes = susquehanna_model.evaluateMC(opt_met = 0)

with open(output_filename, "wb") as f:
    pickle.dump(outcomes, f)
print(outcomes[decision])

