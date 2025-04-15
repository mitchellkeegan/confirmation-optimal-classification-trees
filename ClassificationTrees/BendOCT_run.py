from BendOCT import BendOCTWrapper
from DataUtils import valid_datasets

initial_cuts = {'Subproblem LP': {'Enabled': False},
                'Subproblem Dual Inspection': {'Enabled': False},
                'EQP Chain': {'Enabled': False,
                              'Features Removed': 2,
                              'Disaggregate Alpha': True,
                              'Lazy': 0},
                'EQP Basic': {'Enabled': False,
                              'Features Removed': 0},
                'EQP Target': {'Enabled': False},
                'EQP Chain Alt1': {'Enabled': True,
                                   'Disaggregate': True,
                                   'Features Removed': 3},
                'EQP Chain Alt2': {'Enabled': False,
                                   'Recursive': True,
                                   'Disaggregate': False,
                                   'Features Removed': 1},
                'No Feature Reuse': {'Enabled': False}}

callback_settings = {'Enabled': True,
                     'Enhanced Cuts': False,
                     'D2Subtrees Primal Heuristic': False,
                     'EQP Chain Cutting Planes': False,
                     'Primal Heuristic Cutting Planes': False,
                     'PH CP Symmetric Cuts': False}

opt_params = {'Results Directory': 'test',
              'Initial Cuts': initial_cuts,
              'Callback': callback_settings,
              'Warmstart': True,
              'Polish Warmstart': True,
              'Use Baseline': False,
              'Encoding Scheme': None,
              'Number Buckets': 5}

gurobi_params = {'TimeLimit': 3600,
                 'LogToConsole': 0,
                 'Method': -1}

seeds_values = [0, 42, 579, 2454, 16, 88888]


# TODO: Allow for non-iterables to be inputted as hyperparameters
hyperparameters = {'depth': [1,2,3,4],
                   'EQP Chain Alt1-Disaggregate': [True],
                   'EQP Chain Alt1-Features Removed': [1, 2],
                   'Encoding Scheme': ['Quantile Thresholds', 'Quantile Buckets']}
# datasets = valid_datasets['numerical']
datasets = valid_datasets['mixed']
# datasets = valid_datasets['categorical']

# datasets = ['']

BendOCTWrapper(hyperparameters,datasets,opt_params=opt_params,gurobi_params=gurobi_params)