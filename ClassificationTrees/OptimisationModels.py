import os
import time
import itertools
import csv
from abc import ABC, abstractmethod

from gurobipy import *

from Utils import Tree

class InitialCut(ABC):
    @abstractmethod
    def add_cuts(self,model,data):
        pass

    def UpdateSettings(self, opts, model_opts=None,instance=None):
        if not opts['Enabled']:
            settings_valid = True
            settings_useful = False
        else:
            settings_valid = self.validate_settings(opts,model_opts=model_opts,instance=instance)
            settings_useful = self.useful_settings(opts,model_opts=model_opts,instance=instance)

        if settings_valid:
            self.opts = opts

        return settings_valid, settings_useful

    def gen_CompleteSolution(self):
        # If the initial cut requires auxiliary decision variables it can sometimes be
        # computationally expensive for Gurobi to infer these variables from a partial solution.
        # Use this to complete solution with where \in {'Warm Start', 'Callback'}

        def CompleteSolution(model, data, soln, where):
            pass

        return CompleteSolution

    def useful_settings(self,opts, model_opts=None, instance=None):
        # Overwrite to detect when having these settings enabled with the given model_opts and instance
        # is not useful. i.e. results will be identical to not using the cut or using another version of the cut
        return True

    def validate_settings(self,opts, model_opts=None, instance=None):
        # Overwrite to detect when having these settings enabled with the given model_opts and instance
        # will error out
        return True

class GenCallback(ABC):
    def __init__(self):
        self.callback_name = 'Generic Callback'
        self.default_settings = {'Enabled': False}
        self.opts = self.default_settings

    def gen_callback(self):
        return None

    def update_model(self,model):
        # Can be overwritten to add useful information to model (E.g. a cache used by a primal heuristic)
        pass

    def UpdateSettings(self, opts, model_opts=None, instance=None):
        if not opts['Enabled']:
            settings_valid = True
            settings_useful = False
        else:
            settings_valid = self.validate_settings(opts, model_opts=model_opts, instance=instance)
            settings_useful = self.useful_settings(opts, model_opts=model_opts, instance=instance)

        if settings_valid:
            self.opts = opts

        return settings_valid, settings_useful

    def validate_settings(self, opts, model_opts=None, instance=None):
        return True

    def useful_settings(self, opts, model_opts=None, instance=None):
        return False

class OCT():
    def __init__(self,opt_params, gurobi_params):
        self.GurobiLogFile = ""
        self.opt_params = opt_params
        self.gurobi_params = gurobi_params
        self.model_type = 'OCT'
        self.callback = None

    def SetGurobiLogFile(self,file):
        self.GurobiLogFile = file

    def create_model(self):
        model = Model()

        model.Params.LogToConsole = self.gurobi_params['LogToConsole']

        # By default, the file will be an empty string. Need to set by the SetGurobiLogFile method if logging required
        model.params.LogFile = self.GurobiLogFile
        # Clear out the logfile if it already exists
        if os.path.exists(model.params.LogFile):
            open(model.params.LogFile,'w').close()

        model.Params.MIPGap = self.gurobi_params['MIPGap']
        model.Params.MIPFocus = self.gurobi_params['MIPFocus']
        model.Params.Heuristics = self.gurobi_params['Heuristics']
        model.Params.TimeLimit = self.gurobi_params['TimeLimit']
        model.Params.Threads = self.gurobi_params['Threads']
        model.Params.Method = self.gurobi_params['Method']
        model.Params.NodeMethod = self.gurobi_params['NodeMethod']
        model.Params.Seed = self.gurobi_params['Seed']



        # Set up attribubtes on model for attaching information
        model._opts = set()
        model._times = {'Heuristics': {},
                        'Initial Cuts': {},
                        'Callback': {}}
        model._nums = {'Heuristics': {},
                       'Initial Cuts': {},
                       'Callback': {}}
        model._model_name = self.model_type

        # Do this to have confidence we can access params before we optimise the model (usually from initial cuts)
        model.update()

        return model

    def build_model(self, data):
        build_start_time = time.time()

        model = self.create_model()
        model._data = data
        model._tree = Tree(self.opt_params['depth'])
        model._lambda = self.opt_params.get('lambda', None)
        self.add_vars(model,data)
        self.add_constraints(model,data)
        self.add_objective(model,data)
        self.add_initial_cuts(model, data)
        if self.opt_params['Warmstart']:
            self.warm_start(model,data)

        build_end_time = time.time()
        model._build_time = build_end_time - build_start_time

        self.model = model

    def warm_start(self,model,data):
        pass

    def optimize_model(self):
        self.model.optimize(self.callback)

    def post_process_model(self,data):
        status = self.model.Status

        # Check that model actually solved
        if status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            self.model._log_message = 'Model not solved'
            return

        vars_readable = self.vars_to_readable(self.model, data)
        self.save_model_output(vars_readable, data)
        self.model._log_message = self.output_log_message(vars_readable, data)

    def SetInitialCuts(self,available_cuts):
        self.available_cuts = available_cuts

    def SetCallback(self, callback_generator):
        callback_generator.update_model(self.model)
        self.callback = callback_generator.gen_callback()

    def add_initial_cuts(self,model,data):

        solution_completers = []

        for cut in self.available_cuts.values():
            if cut.opts['Enabled']:
                cut.add_cuts(model, data)
                solution_completers.append(cut.gen_CompleteSolution())

        # Attach functions generated by enabled initial cuts for completing partial solutions
        model._solution_completers = solution_completers

    def save_model_output(self,vars,data):
        return None

    def vars_to_readable(self,data):
        return None

    def output_log_message(self, vars, data):
        return ''