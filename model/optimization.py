from susquehanna_model import susquehannaModel

import pandas as pd
import glob
import tqdm
import numpy as np
import pickle
import os

from ema_workbench import RealParameter, ScalarOutcome, SequentialEvaluator
from ema_workbench import Model
from ema_workbench import Constraint


inputs = 4 # inputs
outputs = 4 # outputs - just the downstream discharge outputs. 
RBFs = inputs+2 # RBFs
n_years = 100

# Initialize the object
susquehanna_model = susquehannaModel(108.5, 505.0, 1, n_years)

susquehanna_model.load_data()

susquehanna_model.setRBF(RBFs, inputs, outputs)

from ema_workbench.em_framework.optimization import GenerationalBorg, HyperVolume, EpsilonProgress

## Optimizing using the workbench


model = Model("SusquehannaModel", function = susquehanna_model.evaluateMC)

model.outcomes = [    ScalarOutcome('hydropower_revenue' , kind = ScalarOutcome.MINIMIZE, expected_range=(0,100))
                    , ScalarOutcome('atomic_power_plant_discharge', kind = ScalarOutcome.MINIMIZE, expected_range=(0,1))
                    , ScalarOutcome('baltimore_discharge', kind = ScalarOutcome.MINIMIZE, expected_range=(0,1)) 
                    , ScalarOutcome('chester_discharge', kind = ScalarOutcome.MINIMIZE , expected_range=(0,1))
                    , ScalarOutcome('recreation', kind = ScalarOutcome.MINIMIZE , expected_range=(0,1))
                    , ScalarOutcome('environment', kind = ScalarOutcome.MINIMIZE , expected_range=(-1,0))
                    , ScalarOutcome('flood_risk', kind = ScalarOutcome.MINIMIZE , expected_range=(0,100))
                    , ScalarOutcome('flood_duration', kind = ScalarOutcome.MINIMIZE , expected_range=(0,100))]

convergence_metrics = [HyperVolume.from_outcomes(model.outcomes),
                       EpsilonProgress()]

constraints = [Constraint("atomic_power_plant_discharge", outcome_names="atomic_power_plant_discharge",
                          function=lambda x:max(0, 0.9 - (-1*x))), 
                          Constraint("flood_risk", outcome_names = "flood_risk", function = lambda x:max(0, x-8.8))]


radii = [RealParameter(f"r{i}", 0,1) for i in range(RBFs*(RBFs-inputs))]
centres = [RealParameter(f"c{i}", -1,1) for i in range(RBFs*(RBFs-inputs))]
weights = [RealParameter(f"w{i}", 0,1) for i in range(RBFs*inputs)]
phaseshift = [RealParameter(f"ps{i}", 0,2*np.pi) for i in range(2)]

model.levers = centres + radii + weights + phaseshift

with SequentialEvaluator(model) as evaluator:
    optimization_results, convergence = evaluator.optimize(searchover='levers', 
                        nfe=1000, epsilons=[0.5,0.05, 0.05, 0.05, 0.05, 0.001, 0.001, 0.001], algorithm=GenerationalBorg,convergence=convergence_metrics)