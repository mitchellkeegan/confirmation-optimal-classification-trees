from FlowOCT import FlowOCTWrapper
from DataUtils import valid_datasets

initial_cuts = {'EQP Basic': {'Enabled': False,
                              'Features Removed': 1},
                'No Feature Reuse': {'Enabled': False},
                'EQP Target': {'Enabled': False}}

callback_settings = {'Enabled': True,
                     'D2Subtrees Primal Heuristic': False,
                     'Primal Heuristic Cutting Planes': True,
                     'EQP Basic Cutting Planes': False}

opt_params = {'Results Directory': 'PrimalHeuristicCuttingPlaneBenchmark',
              'Initial Cuts': initial_cuts,
              'Callback': callback_settings,
              'Warmstart': True,
              'Polish Warmstart': True,
              'Use Baseline': False,
              'Encoding Scheme': None}

gurobi_params = {'TimeLimit': 3600,
                 'Threads': 1,
                 'MIPGap': 0,
                 'Method': -1,
                 'LogToConsole': 0}

hyperparameters = {'depth': [4]}

datasets = valid_datasets['categorical']
# datasets = ['kr-vs-kp']

FlowOCTWrapper(hyperparameters,datasets,opt_params=opt_params,gurobi_params=gurobi_params)