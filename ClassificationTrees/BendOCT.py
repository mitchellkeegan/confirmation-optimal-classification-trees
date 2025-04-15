import os
import sys
import itertools
import time
import csv

from OptimisationModels import OCT, InitialCut, GenCallback
from Utils import CART_Heuristic, Equivalent_Points, logger, optimise_subtrees, optimise_depth2_subtree, find_split_sets
from DataUtils import load_instance, valid_datasets


import numpy as np
from gurobipy import *

name_dict = {'Encoding Scheme': 'ES',
             'Quantile Buckets': 'QB',
             'Quantile Thresholds': 'QT',
             'Features Removed': 'FR',
             'Disaggregate': 'DA',
             'True': 'T',
             'False': 'F',
             'Disaggregate Alpha': 'DA',
             'EQP Basic': 'EQPB',
             'EQP Chain': 'EQPC',
             'EQP Chain Alt1': 'EQPCA',
             'Primal Heuristic Cutting Planes': 'PH Cuts'}

class BendOCT(OCT):
    def __init__(self,opt_params, gurobi_params):

        super().__init__(opt_params, gurobi_params)
        self.model_type = 'BendOCT'

    def add_vars(self,model,data):
        I = data['I']
        F = data['F']
        K = data['K']

        tree = model._tree

        b = {(n, f): model.addVar(vtype=GRB.BINARY, name=f'b_{n}{f}')
                  for n in tree.B for f in F}
        w = {(k, n): model.addVar(vtype=GRB.CONTINUOUS, name=f'w_{k}^{n}')
                  for k in K for n in tree.L}
        theta = {i: model.addVar(vtype=GRB.CONTINUOUS, ub=1, name=f'theta_{i}')
                      for i in I}

        model._variables = {'b': b,
                            'w': w,
                            'theta': theta}

    def add_constraints(self,model,data):
        F = data['F']
        K = data['K']

        b = model._variables['b']
        w = model._variables['w']

        tree = model._tree

        # Can only branch on one variable at each branch node
        only_one_branch = {n: model.addConstr(quicksum(b[n, f] for f in F) == 1)
                           for n in tree.B}

        # Make a single class prediction at each leaf node
        leaf_prediction = {n: model.addConstr(quicksum(w[k, n] for k in K) == 1)
                           for n in tree.L}

    def add_objective(self,model,data):

        theta = model._variables['theta']
        I = data['I']
        weights = data['weights']

        model.setObjective(quicksum(weights[i] * theta[i] for i in I), GRB.MAXIMIZE)

    def warm_start(self,model,data):

        compressed = data['compressed']
        tree = model._tree

        if self.opt_params['Polish Warmstart']:
            model._opts.add('CART polish solutions')

        if compressed:
            X, y = data['Xf'], data['yf']
        else:
            X, y = data['X'], data['y']

        b = model._variables['b']
        w = model._variables['w']
        theta = model._variables['theta']

        heuristic_start_time = time.time()

        HeuristicSoln = CART_Heuristic(X, y, tree, model._opts,
                                       cat_feature_maps=data['Categorical Feature Map'],
                                       num_feature_maps=data['Numerical Feature Map'])

        if HeuristicSoln is not None:
            for k, v in HeuristicSoln['b'].items():
                b[k].Start = v

            for k, v in HeuristicSoln['w'].items():
                w[k].Start = v

            for i, v in enumerate(HeuristicSoln['theta']):
                if compressed:
                    idx_map = self.instance_data['idxf_to_idxc']
                    j = idx_map[i]
                    theta[j].Start = v
                else:
                    theta[i].Start = v

            for sc in model._solution_completers:
                sc(model, data, HeuristicSoln, 'Warm Start')

            if 'theta old' in HeuristicSoln:
                print(
                    f'CART returned Heuristic Solution with {sum(HeuristicSoln['theta'])}/{len(y)} samples classified '
                    f'(polished from {sum(HeuristicSoln['theta old'])}) correctly in {time.time() - heuristic_start_time:.2f}s')
            else:
                print(
                    f'CART returned Heuristic Solution with {sum(HeuristicSoln['theta'])}/{len(y)} samples classified '
                    f'correctly in {time.time() - heuristic_start_time:.2f}s')

        else:
            print('CART did not return a valid heuristic solution')

        model._times['Heuristics']['CART'] = time.time() - heuristic_start_time
        model._nums['Heuristics']['CART'] = sum(HeuristicSoln['theta'])

class BendersCallback(GenCallback):
    def __init__(self):
        self.callback_name = 'Standard Benders Cuts'
        self.default_settings = {'Enabled': True,
                                 'Enhanced Cuts': False,
                                 'D2Subtrees Primal Heuristic': False,
                                 'EQP Chain Cutting Planes': False,
                                 'Primal Heuristic Cutting Planes': False,
                                 'PH CP Symmetric Cuts': False}
        self.opts = self.default_settings

    def update_model(self,model):

        model.Params.LazyConstraints = 1

        if self.opts['Enhanced Cuts']:
            b = model._variables['b']
            theta = model._variables['theta']

            for k in theta:
                theta[k].vtype = GRB.BINARY

            for k in b:
                b[k].BranchPriority = 1

        if self.opts['Primal Heuristic Cutting Planes']:
            model.Params.PreCrush = 1

            model._nums['Callback']['PH Cutting Planes'] = 0
            model._times['Callback']['PH Cutting Planes'] = 0.0


        if self.opts['EQP Chain Cutting Planes']:

            # model.Params.Presolve = 0
            model.Params.PreCrush = 1
            # model.Params.Presolve = 0
            # model.Params.CutPasses = 10000
            # model.Params.Cuts = 3

            data = model._data

            X = data['X']
            y = data['y']
            eqp_cuts = Equivalent_Points(X,y,max_removed=0)

            alpha = {}

            # for cut_idx, _, _ in eqp_cuts:
            #     alpha[cut_idx] = model.addVar(vtype=GRB.CONTINUOUS)
            #
            # model._variables['alpha'] = alpha
            model._CallbackStorage = {'EQP Cuts': eqp_cuts}

            model._nums['Callback']['EQP Cutting Planes'] = 0
            model._times['Callback']['EQP Cutting Planes'] = 0.0

        if self.opts['Primal Heuristic Cutting Planes'] or self.opts['D2Subtrees Primal Heuristic']:
            model._D2SubtreeCache = {}

    def gen_callback(self):

        if not self.opts['Enabled']:
            return None

        nodes_seen = set()
        EPS = 1e-4

        def DFS(root, I, bV, tree, F, X, cut_vars=False, changed_root_branch=None):
            # INPUTS
            # root - root node to begin DFS from
            # I - subset samples on which to run search
            # bV - branch decision variables to use
            # cut_vars - True -> keep track of branch variables which would have sent each sample onto a different leaf
            # changed_root_branch - Substituted branch variable for root node. Used in Benders cuts based on all samples

            # RETURNS
            # sample_leaf_assigments -
            # sample_in_node -
            # cut_node_branch_feature -
            # cut_branch_vars -

            subtree_branch_nodes, subtree_leaf_nodes = tree.descendants(root, split_nodes=True)
            node_branch_feature = {}

            for n in subtree_branch_nodes:
                for f in F:
                    if bV[n, f] > 0.5:
                        node_branch_feature[n] = f

            if changed_root_branch is not None:
                node_branch_feature[root] = changed_root_branch

            sample_node_path = {i: [] for i in I}
            samples_in_node = {n: [] for n in subtree_branch_nodes + subtree_leaf_nodes}

            cut_branch_vars = {i: [] for i in I}

            # Run a DFS from the root down to the leaves for each sample
            for i in I:
                sample = X[i, :]

                current_node = root
                while current_node in tree.B:
                    sample_node_path[i].append(current_node)
                    samples_in_node[current_node].append(i)
                    branch_feature = node_branch_feature[current_node]

                    if sample[branch_feature] == 0:
                        # Sample branches to the left
                        # Find features that would have sent the sample down the right branch if branched on
                        if cut_vars:
                            for f in F:
                                if sample[f] == 1:
                                    cut_branch_vars[i].append((current_node, f))

                        current_node = tree.left_child(current_node)

                    else:
                        # Sample branches to the right
                        # Find features that would have sent the sample down the left branch if branched on
                        if cut_vars:
                            for f in F:
                                if sample[f] == 0:
                                    cut_branch_vars[i].append((current_node, f))

                        current_node = tree.right_child(current_node)

                # Sample i now in a leaf node
                samples_in_node[current_node].append(i)
                sample_node_path[i].append(current_node)

            return sample_node_path, samples_in_node, node_branch_feature, cut_branch_vars

        def callback(model, where):

            data = model._data
            tree = model._tree

            if where == GRB.Callback.MIPSOL:

                callback_start_time = time.time()

                I = data['I']
                F = data['F']
                X = data['X']
                y = data['y']
                weights = data['weights']
                cat_feature_maps = data['Categorical Feature Map']
                num_feature_maps = data['Numerical Feature Map']

                b = model._variables['b']
                w = model._variables['w']
                theta = model._variables['theta']

                bV = model.cbGetSolution(b)
                wV = model.cbGetSolution(w)
                thetaV = model.cbGetSolution(theta)

                sample_node_path, samples_in_node, node_branch_feature, cut_branch_vars = DFS(1, I, bV, tree, F, X, cut_vars=True)

                for i in I:
                    leaf_node = sample_node_path[i][-1]

                    if wV[y[i], leaf_node] < thetaV[i] - EPS:

                        parent = tree.parent(leaf_node)
                        other_leaf = [n for n in tree.children(parent) if n != leaf_node][0]
                        if self.opts['Enhanced Cuts'] and wV[y[i], other_leaf] < thetaV[i] - EPS:
                            tCon = (theta[i] <= quicksum(b[n, f] for n, f in cut_branch_vars[i] if n != parent) + w[y[i], leaf_node]
                                    + (quicksum(b[n,f] for n,f in cut_branch_vars[i] if n == parent) + w[y[i],other_leaf])/2)
                        else:
                            tCon = (theta[i] <= quicksum(b[n, f] for n, f in cut_branch_vars[i]) + w[y[i], leaf_node])
                        model.cbLazy(tCon)

                        if 'Standard Benders' not in model._nums['Callback']:
                            model._nums['Callback']['Standard Benders'] = 1
                        else:
                            model._nums['Callback']['Standard Benders'] += 1

                callback_runtime = time.time() - callback_start_time
                if 'Standard Benders' not in model._times['Callback']:
                    model._times['Callback']['Standard Benders'] = callback_runtime
                else:
                    model._times['Callback']['Standard Benders'] += callback_runtime

                if self.opts['D2Subtrees Primal Heuristic'] and tree.depth > 1:
                    if not model._data['compressed']:

                        if 'Polish Solutions' not in model._nums['Callback']:
                            model._nums['Callback']['Polish Solutions'] = 0
                        if 'Polish Solutions' not in model._times['Callback']:
                            model._times['Callback']['Polish Solutions'] = 0.0

                        polish_soln_start_time = time.time()

                        b_subtrees, w_subtrees, theta_polished = optimise_subtrees(X, y, samples_in_node, tree, model._opts, node_branch_feature,
                                                                                   cache=model._D2SubtreeCache,
                                                                                   weights=weights,
                                                                                   cat_feature_maps=cat_feature_maps,
                                                                                   num_feature_maps=num_feature_maps)

                        if b_subtrees is not None:
                            PossObj = sum(theta_polished)
                            CurrObj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                            if PossObj > CurrObj + 0.1:
                                bV |= b_subtrees
                                wV |= w_subtrees
                                thetaV = theta_polished

                                # for cut_idx, rhs_bound, removed_features in model._eqp_cuts:
                                #     if sum(thetaV[idx] for idx in cut_idx)

                                model.cbSetSolution(b, bV)
                                model.cbSetSolution(w, wV)
                                model.cbSetSolution(theta, thetaV)

                                for sc in model._solution_completers:
                                    sc(model, data, {'b': bV}, 'Callback')

                                model.cbUseSolution()


                                model._nums['Callback']['Polish Solutions'] += 1
                                print(f'**** Callback Primal Heuristic improved solution from {CurrObj} to {PossObj} ****')

                        model._times['Callback']['Polish Solutions'] += time.time() - polish_soln_start_time

                    else:
                        print('Solution polishing has not been tested with compressed datasets\n'
                              'Must manually remove check in callback to enable it')

            if where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                if self.opts['EQP Chain Cutting Planes']:
                    node_count = model.cbGet(GRB.Callback.MIPNODE_NODCNT)

                    cut_start_time = time.time()

                    if node_count >= 0:

                        data = model._data
                        CallbackStorage = model._CallbackStorage

                        theta = model._variables['theta']
                        thetaR = model.cbGetNodeRel(theta)

                        eqp_cuts = CallbackStorage['EQP Cuts']

                        for cut_idx, rhs_bound, removed_features in eqp_cuts:
                            if len(removed_features) == 0:
                                if sum(thetaR[i] for i in cut_idx) > rhs_bound + EPS:
                                    tcon = (quicksum(theta[i] for i in cut_idx) <= rhs_bound)
                                    model.cbCut(tcon)

                                    model._nums['Callback']['EQP Cutting Planes'] += 1

                                # model.cbCut(quicksum(theta[i] for i in cut_idx), GRB.LESS_EQUAL, rhs_bound + EPS)

                    model._times['Callback']['EQP Cutting Planes'] += time.time() - cut_start_time

                    # if node_count not in nodes_seen:
                    #     print(node_count)
                    #     nodes_seen.add(node_count)

                if self.opts['Primal Heuristic Cutting Planes']:

                    cut_start_time = time.time()

                    F = data['F']

                    b = model._variables['b']
                    theta = model._variables['theta']

                    bR = model.cbGetNodeRel(b)
                    thetaR = model.cbGetNodeRel(theta)

                    integral_paths = []

                    node_info = [(1, tuple())]

                    while len(node_info) > 0:

                        node, path = node_info.pop()
                        # Check if we terminate with an integral path found
                        if tree.height(node) == 2:
                            integral_paths.append(path)
                            continue

                        # Check if we have an integral branch variable
                        branch_var = None
                        for f in F:
                            if bR[node, f] > 1 - EPS:
                                branch_var = f
                                continue

                        # If we do add its children to node_info
                        if branch_var is not None:
                            left_child, right_child = tree.children(node)
                            node_info.append((left_child, path + ((node, branch_var, 0),)))
                            node_info.append((right_child, path + ((node, branch_var, 1),)))

                    for path in integral_paths:
                        # For each path we want to add a cut on it and any symmetrical paths
                        # Each path has elements (node, branch_var, direction)

                        path = list(path)

                        # First determine which subset of the dataset follows the integral path
                        X = data['X']
                        y = data['y']
                        I = np.asarray(data['I'])
                        F = data['F']
                        weights = data['weights']


                        # I_P = list(I)
                        # for node, branch_var, dir in path:
                        #     I_P = [i for i in I_P if X[i, branch_var] == dir]

                        I_mask = np.ones(len(I),dtype=bool)
                        for _, branch_var, dir in path:
                            I_mask &= (X[:,branch_var] == dir)

                        I_P = I[I_mask]

                        if len(I_P) == 0:
                            continue

                        path_key = frozenset((branch_var, dir) for _, branch_var, dir in path)

                        if path_key in model._D2SubtreeCache:
                            # print('Avoided double computation')
                            _,_,theta_idx = model._D2SubtreeCache[path_key]
                        else:
                            # Find optimal depth 2 subtree for subset I_P
                            b_subtree, w_subtree, theta_idx = optimise_depth2_subtree(X[I_P, :],y[I_P])

                            if theta_idx is None:
                                # subroutine should return None if something went wrong
                                continue
                            else:
                                # Update the cache with the calculated subtree optimal soln
                                model._D2SubtreeCache[path_key] = (b_subtree, w_subtree, theta_idx)

                        # Search over each node in the path and determine if, given the other predicates, we can add extra variables to the cut
                        # updated_path = []
                        # for i in range(len(path)):
                        #     partial_path = path[:i] + path[i + 1:]
                        #
                        #     I_mask = np.ones(len(I), dtype=bool)
                        #     for _, branch_var, dir in partial_path:
                        #         I_mask &= (X[:, branch_var] == dir)
                        #
                        #     X_filt = X[I[I_mask], :]
                        #
                        #     n, branch_var, dir = path[i]
                        #
                        #     # We want to find any features (columns) for which all samples in X_filt have the same values as the target_col
                        #     target_col = np.expand_dims(X_filt[:, branch_var], axis=1)
                        #     matched_cols = np.all(X_filt == target_col, axis=0)
                        #     branch_var_list = np.nonzero(matched_cols)[0].tolist()
                        #
                        #     # Update the path
                        #     updated_path.append((n, branch_var_list, dir))
                        #
                        # path = updated_path

                        # subroutine returns indices of I_P which were correctly classified. Take len to get optimal objective
                        optimal_subtree_obj = len(theta_idx)

                        # Check if the upper bound found is violated by the relaxation solution
                        if sum(thetaR[i] for i in I_P) > optimal_subtree_obj + EPS:

                            if self.opts['PH CP Symmetric Cuts']:
                                permuted_paths = itertools.permutations(path)
                            else:
                                permuted_paths = [path]

                            # If the cut is violated, calculate the upper bound on the samples
                            # Then apply the cut to all symmetric paths

                            # Get all permutations of branch variables and orderings
                            # Each one corresponds to a possible path through the tree
                            for path_permutation in permuted_paths:

                                # print(path_permutation)

                                node = 1
                                path_branch_choices = []

                                for _, branch_var_list, dir in path_permutation:
                                    path_branch_choices.append((node, branch_var_list))
                                    if dir == 0:
                                        node = tree.left_child(node)
                                    elif dir == 1:
                                        node = tree.right_child(node)


                                relaxing_branch_vars = quicksum(quicksum(b[n, ff] for ff in F if ff != var_list) for n, var_list in path_branch_choices)
                                rhs = optimal_subtree_obj + (len(I_P) - optimal_subtree_obj) * relaxing_branch_vars
                                model.cbLazy(quicksum(theta[i] for i in I_P) <= rhs)

                                model._nums['Callback']['PH Cutting Planes'] += 1

                            # left_vars, right_vars = [], []
                            # for _, branch_var, dir in path:
                            #     if dir == 0:
                            #         left_vars.append(branch_var)
                            #     else:
                            #         right_vars.append(branch_var)
                            #
                            # dirs_taken = ['l'] * len(left_vars) + ['r'] * len(right_vars)
                            #
                            # for dir_permutation in set(itertools.permutations(dirs_taken)):
                            #     left_nodes, right_nodes = [], []
                            #     n = 1
                            #     for dir in dir_permutation:
                            #         if dir == 'l':
                            #             left_nodes.append(n)
                            #             n = tree.left_child(n)
                            #         elif dir == 'r':
                            #             right_nodes.append(n)
                            #             n = tree.right_child(n)
                            #         else:
                            #             assert False
                            #
                            #     relaxing_vars_left = quicksum(b[n, f] for n in left_nodes for f in F if f not in left_vars)
                            #     relaxing_vars_right = quicksum(b[n, f] for n in right_nodes for f in F if f not in right_vars)
                            #
                            #     rhs = optimal_subtree_obj + (len(I_P) - optimal_subtree_obj) * (relaxing_vars_left + relaxing_vars_right)
                            #     model.cbLazy(quicksum(theta[i] for i in I_P) <= rhs)
                            #
                            #     model._nums['Callback']['PH Cutting Planes'] += 1

                    model._times['Callback']['PH Cutting Planes'] += time.time() - cut_start_time

        return callback

    def useful_settings(self, opts, model_opts=None, instance=None):
        if opts['D2Subtrees Primal Heuristic'] or opts['EQP Chain Cutting Planes'] or opts['Primal Heuristic Cutting Planes']:
            return True
        else:
            return False

    def validate_settings(self,opts, model_opts=None, instance=None):
        if opts['Primal Heuristic Cutting Planes'] or opts['D2Subtrees Primal Heuristic']:
            if instance in valid_datasets['numerical'] + valid_datasets['mixed']:
                if model_opts['Encoding Scheme'] in ['Bucketisation', 'Full']:
                    print('\nD2Subtrees Primal Heuristic is too slow with numerical features with threshold encodings\n')
                    return False

        return True

class SubproblemLP(InitialCut):
    def __init__(self):
        self.cut_name = 'Subproblem LP'
        self.default_settings = {'Enabled': False}
        self.opts = self.default_settings

    def add_cuts(self,model,data):

        tree = model._tree
        theta = model._variables['theta']
        b = model._variables['b']
        w = model._variables['w']

        X = data['X']
        y = data['y']
        I = data['I']
        F = data['F']
        weights = data['weights']

        EPS = 1e-4

        # Relax the MP
        for k in b:
            b[k].vtype = GRB.CONTINUOUS

        cut_start_time = time.time()
        num_iterations = 1
        LogToConsoleSetting = model.Params.LogToConsole
        LogFile = model.Params.LogFile
        model.Params.LogFile = ''
        model.Params.LogToConsole = 0
        model._nums['Initial Cuts']['Subproblem LP'] = 0

        while True:
            model.optimize()
            cuts_added = 0
            for i in I:
                # Solve the i^th dual subproblem
                # Use env to suppress any output from LP solve
                with Env(empty=True) as env:
                    env.setParam('OutputFlag',0)
                    env.start()
                    with Model(env=env) as LP_model:
                        LP_model.Params.LogToConsole = 0

                        z = {(n1, n2): LP_model.addVar(vtype=GRB.CONTINUOUS, name=f'z_{n1}{n2}')
                             for n1 in tree.B for n2 in tree.children(n1)}

                        z[tree.source, 1] = LP_model.addVar(vtype=GRB.CONTINUOUS)
                        for n in tree.L:
                            z[n, tree.sink] = LP_model.addVar(vtype=GRB.CONTINUOUS)

                        # Flow from source at most one
                        source_flow_bound = LP_model.addConstr(z[tree.source, 1] <= 1)

                        # Flow in = flow out at branch nodes
                        branch_flow_equality = {n: LP_model.addConstr(
                            z[tree.parent(n), n] == quicksum(z[n, n_child] for n_child in tree.children(n)))
                                                for n in tree.B}

                        # Flow in = flow out at leaf nodes
                        leaf_flow_equality = {n: LP_model.addConstr(z[tree.parent(n), n] == z[n, tree.sink])
                                              for n in tree.L}

                        # Bound the left child flow capacity at each branch node
                        left_child_capacity = {n: LP_model.addConstr(z[n, tree.left_child(n)] <=
                                                                     quicksum(b[n, f].X for f in F if X[i, f] < 0.5))
                                               for n in tree.B}

                        # Bound the left child flow capacity at each branch node
                        right_child_capacity = {n: LP_model.addConstr(z[n, tree.right_child(n)] <=
                                                                      quicksum(b[n, f].X for f in F if X[i, f] > 0.5))
                                                for n in tree.B}

                        # Set capacity of edges from leaves to sink node
                        sink_flow_bound = {n: LP_model.addConstr(z[n, tree.sink] <= w[y[i], n].X)
                                           for n in tree.L}

                        LP_model.setObjective(weights[i] * quicksum(z[(n, tree.sink)] for n in tree.L), GRB.MAXIMIZE)

                        LP_model.optimize()

                        if theta[i].X > LP_model.objVal + EPS:
                            if (LP_model.objVal > 0.05) and (max([left_child_capacity[n].Pi for n in tree.B] + [right_child_capacity[n].Pi for n in tree.B]) < 0.95):
                                print(5)
                            cuts_added += 1
                            model.addConstr(theta[i] <= quicksum(
                                left_child_capacity[n].Pi * b[n, f] for n in tree.B for f in F if X[i, f] == 0)
                                            + quicksum(
                                right_child_capacity[n].Pi * b[n, f] for n in tree.B for f in F if X[i, f] == 1)
                                            + quicksum(sink_flow_bound[n].Pi * w[y[i], n] for n in tree.L))

            model._nums['Initial Cuts']['Subproblem LP'] += cuts_added

            if cuts_added == 0:
                print(f'Subproblem LP added {model._nums['Initial Cuts']['Subproblem LP']} cuts in {num_iterations} iterations')
                break

            num_iterations += 1

        # Unrelax the MP
        for k in b:
            b[k].vtype = GRB.BINARY

        model.Params.LogToConsole = LogToConsoleSetting
        model.Params.LogFile = LogFile
        model._times['Initial Cuts']['Subproblem LP'] = time.time() - cut_start_time

class SubproblemDualInspection(InitialCut):
    def __init__(self):
        self.cut_name = 'Subproblem Dual Inspection'
        self.default_settings = {'Enabled': False}
        self.opts = self.default_settings

    def add_cuts(self,model,data):

        tree = model._tree
        theta = model._variables['theta']
        b = model._variables['b']
        w = model._variables['w']

        X = data['X']
        y = data['y']
        I = data['I']
        F = data['F']
        weights = data['weights']

        EPS = 1e-4

        # Relax the MP
        for k in b:
            b[k].vtype = GRB.CONTINUOUS

        cut_start_time = time.time()
        num_iterations = 1
        LogToConsoleSetting = model.Params.LogToConsole
        LogFile = model.Params.LogFile
        model.Params.LogFile = ''
        model.Params.LogToConsole = 0
        model._nums['Initial Cuts']['Subproblem Dual Inspection'] = 0

        while True:
            model.optimize()
            cuts_added = 0
            for i in I:
                ##### Solve by inspection #####

                # Fill in flow graph capacities
                cap = {}
                cap[tree.source, 1] = 1
                for n in tree.B:
                    cap[n, tree.left_child(n)] = sum(b[n, f].X for f in F if X[i, f] == 0)
                    cap[n, tree.right_child(n)] = sum(b[n, f].X for f in F if X[i, f] == 1)
                for n in tree.L:
                    cap[n, tree.sink] = w[y[i], n].X

                node_info = {n: [] for n in tree.B + tree.L}

                for n in tree.L:
                    if cap[tree.parent(n), n] < cap[n, tree.sink] + EPS:
                        edge = (tree.parent(n), n)
                    else:
                        edge = (n, tree.sink)

                    node_info[n].append((cap[edge], edge))

                    # node_info[n]['cuts'] = [edge]
                    # node_info[n]['cut capacities'] = cap[edge]

                for n in reversed(tree.B):
                    edge = (tree.parent(n), n)
                    child_min_cuts = node_info[tree.left_child(n)] + node_info[tree.right_child(n)]
                    child_cut_capacity = sum(cut[0] for cut in child_min_cuts)
                    if n > 1 and cap[edge] < child_cut_capacity + EPS:
                        # In this case add a cut from the branch node to the parent
                        node_info[n].append((cap[edge], edge))
                    else:
                        # Otherwise keep the cuts from lower down in the tree
                        node_info[n] = child_min_cuts

                min_cuts = node_info[1]
                min_cut_obj = sum(cut[0] for cut in min_cuts)

                if theta[i].X > min_cut_obj + EPS:
                    # If cuts are violated, add back to MP as Benders cuts
                    left_edges = []
                    right_edges = []
                    sink_edges = []

                    branch_nodes = set(tree.B)
                    leaf_nodes = set(tree.L)
                    for cut_capacity, edge in min_cuts:
                        parent, child = edge
                        if parent in leaf_nodes:
                            # Cut is from a leaf node to the sink
                            sink_edges.append((cut_capacity, parent))
                        elif parent in branch_nodes:
                            # Check if cut is on a left or right edge from parent
                            if child == tree.left_child(parent):
                                left_edges.append((cut_capacity, parent))
                            elif child == tree.right_child(parent):
                                right_edges.append((cut_capacity, parent))
                            else:
                                raise Exception('Invalid edge in min cut set')
                        else:
                            raise Exception('Where did you find this node?')

                    con = (theta[i] <= quicksum(b[n, f] for _, n in left_edges for f in F if X[i, f] == 0)
                           + quicksum(b[n, f] for _, n in right_edges for f in F if X[i, f] == 1)
                           + quicksum(w[y[i], n] for _, n in sink_edges))

                    model.addConstr(con)

                    cuts_added += 1

            model._nums['Initial Cuts']['Subproblem Dual Inspection'] += cuts_added

            if cuts_added == 0:
                print(f'Subproblem LP added {model._nums['Initial Cuts']['Subproblem Dual Inspection']} cuts in {num_iterations} iterations')
                break

            num_iterations += 1

        # Unrelax the MP
        for k in b:
            b[k].vtype = GRB.BINARY

        model.Params.LogToConsole = LogToConsoleSetting
        model.Params.LogFile = LogFile
        model._times['Initial Cuts']['Subproblem Dual Inspection'] = time.time() - cut_start_time

class NoFeatureReuse(InitialCut):
    def __init__(self):
        self.cut_name = 'No Feature Reuse'
        self.default_settings = {'Enabled': False}
        self.opts = self.default_settings

    def validate_settings(self,opts,model_opts=None,instance=None):
        if instance is not None and model_opts is not None:
            depth = model_opts['depth']
            if depth > 4 and instance in ['hayes-roth', 'balance-scale']:
                return False
        return True

    def add_cuts(self, model, data):

        model._opts.add('No Feature Reuse')

        cat_feature_maps = data['Categorical Feature Map']
        num_feature_maps = data['Numerical Feature Map']

        F = data['F']

        tree = model._tree
        b = model._variables['b']

        cut_start_time = time.time()
        cuts_added = 0

        # Check how the numerical features were encoded to determine
        if data['encoding'] in ['Bucketisation', 'Full', 'Quantile Thresholds']:
            thresholded_feature_maps = num_feature_maps
            onehot_feature_maps = cat_feature_maps
            model._opts.add('Threshold Encoding')
        else:
            # If we used a bucket-like encoding then we can treat numerical features like categorical ones
            thresholded_feature_maps = []
            onehot_feature_maps = cat_feature_maps + num_feature_maps

        for Nf in thresholded_feature_maps:
            for n in tree.B:
                left_subtree_root, right_subtree_root = tree.children(n)

                if left_subtree_root in tree.B:
                    left_subtree, _ = tree.descendants(left_subtree_root,split_nodes=True)
                else:
                    left_subtree = []

                if right_subtree_root in tree.B:
                    right_subtree, _ = tree.descendants(right_subtree_root, split_nodes=True)
                else:
                    right_subtree = []

                for n_d in right_subtree:
                    model.addConstr(quicksum((idx+1)*b[n,f] for idx,f in enumerate(Nf)) <=
                                    quicksum(idx * b[n_d,f] for idx,f in enumerate(Nf)) + len(Nf) * quicksum(b[n_d,f] for f in F if f not in Nf))
                    cuts_added += 1

                for n_d in left_subtree:
                    model.addConstr(quicksum((len(Nf) - idx - 1) * b[n, f] for idx, f in enumerate(Nf)) <=
                                    quicksum((len(Nf) - idx) * b[n_d, f] for idx, f in enumerate(Nf)) + len(Nf) * quicksum(
                        b[n_d, f] for f in F if f not in Nf))
                    cuts_added += 1

        # for Nf in num_feature_maps:
        #     for n in tree.B:
        #         path = tree.ancestors(n, branch_dirs=True)
        #         path_left, path_right = [], []
        #         for n_a, d in path.items():
        #             if d == 0:
        #                 path_left.append(n_a)
        #             elif d == 1:
        #                 path_right.append(n_a)
        #             else:
        #                 raise Exception('???')
        #
        #         for f in Nf:
        #             # Cannot use feature f at node n if either:
        #             # 1) Another feature in Nf which represents a smaller threshold was used at an ancestor which branched right
        #             # 2) Another feature in Nf which represents a larger threshold was used at an ancestor which branched left
        #
        #             # First try - disaggregate the constraints
        #
        #             # for n_a in path_left:
        #             #     model.addConstr(b[n,f] <= 1 - quicksum(b[n_a,f_l] for f_l in Nf if f_l <= f))
        #             # for n_a in path_right:
        #             #     model.addConstr(b[n, f] <= 1 - quicksum(b[n_a, f_g] for f_g in Nf if f_g >= f))
        #
        #             # Second option - constant in front quicksum so b[n,f] <= 0 when offending features used in all ancestors)
        #             # Unless none of the offending features are present we get b[n,f] < 1 strictly so b[n,f]=0 for binary b
        #             # Fewer constraint but weaker relaxation in the sense that it only force b[n,f] to zero for b binary, not in the relaxation
        #             model.addConstr(quicksum(b[n_a, f_g] for n_a in path_right for f_g in Nf if f_g >= f) +
        #                             quicksum(b[n_a, f_l] for n_a in path_left for f_l in Nf if f_l <= f)
        #                             <= len(path) * (1 - b[n,f]))
        #
        #             cuts_added += 1


        for Cf in onehot_feature_maps:
            if len(Cf) == 1:
                f = Cf[0]
                for n in tree.L:
                    con_name = f'Leaf={n}_feature={f}'
                    model.addConstr(quicksum(b[n_a,f] for n_a in tree.ancestors(n)) <= 1,
                                    name=con_name)
            else:
                for n in tree.B:
                    right_subtree_root = tree.right_child(n)
                    if right_subtree_root in tree.B:
                        subtree, _ = tree.descendants(right_subtree_root, split_nodes=True)

                        cuts_added += 1
                        con_name = f'Root={n}_Group=[' + ','.join(str(f) for f in Cf) + ']'
                        if len(con_name) > 255:
                            print(f'Categorical NoFeatureReuse constraint given name with {len(con_name)} characters (too long). Potential issue with dataset encoding')
                            con_name = ''
                        # TODO: Could tighten this but I don't think it's actually very important to do so
                        model.addConstr(quicksum(b[n_d,f] for n_d in subtree for f in Cf) <= len(subtree) * (1 - quicksum(b[n,f] for f in Cf)),
                                        name=con_name)

        model._times['Initial Cuts']['No Feature Reuse'] = 0.0
        model._nums['Initial Cuts']['No Feature Reuse'] = cuts_added

class EQPBasic(InitialCut):
    def __init__(self):
        self.cut_name = 'EQP Basic'
        self.default_settings = {'Enabled': False,
                                 'Features Removed': 0}
        self.opts = self.default_settings

    def validate_settings(self,opts, model_opts=None, instance=None):
        features_removed = opts['Features Removed']

        # Check if new settings are valid for the new opts
        if features_removed not in [0, 1, 2]:
            return False

        return True

    def useful_settings(self,opts,model_opts=None,instance=None):
        features_removed = opts['Features Removed']

        if instance is not None:
            eqp_datasets = valid_datasets[f'eqp{features_removed}']
            if instance in valid_datasets['categorical']:
                eqp_encoded_datasets = eqp_datasets['categorical']
            elif instance in valid_datasets['numerical']:
                encoding_scheme = model_opts['Encoding Scheme'] + '-' + str(model_opts['Number Buckets'])
                eqp_encoded_datasets = eqp_datasets['numerical'][encoding_scheme]
            elif instance in valid_datasets['mixed']:
                encoding_scheme = model_opts['Encoding Scheme'] + '-' + str(model_opts['Number Buckets'])
                eqp_encoded_datasets = eqp_datasets['mixed'][encoding_scheme]

        if instance not in eqp_encoded_datasets:
            return False

        return True

    def add_cuts(self,model,data):
        I = data['I']
        F = data['F']
        X = data['X']
        y = data['y']

        tree = model._tree

        b = model._variables['b']
        theta = model._variables['theta']

        cut_start_time = time.time()

        max_removed = self.opts['Features Removed']


        eqp_cuts = find_split_sets(data, max_removed=max_removed)

        data['EQP Cuts'] = eqp_cuts

        if max_removed >= 1:
            # For practical reasons beta is indexed by tuples which allows for generalisation to betas with multiple features
            beta = {(f,): model.addVar(vtype=GRB.CONTINUOUS, name=f'beta_{f}')
                    for f in F}
            link_betas = {f: model.addConstr(beta[(f,)] <= quicksum(b[n, f] for n in tree.B), name=f'link_beta_{f}')
                          for f in F}
            model._variables['beta'] = beta

        for cut_idx, rhs_bound, removed_features in eqp_cuts:
            if len(removed_features) == 0:
                rhs = rhs_bound
            else:
                # Cover case when multiple features removed
                if removed_features not in beta:
                    beta[removed_features] = model.addVar(vtype=GRB.CONTINUOUS)
                    model.addConstr(beta[removed_features] <= quicksum(beta[(f,)] for f in removed_features))

                rhs = rhs_bound + (len(cut_idx) - rhs_bound) * beta[removed_features]

            model.addConstr(quicksum(theta[i] for i in cut_idx) <= rhs)

        model._times['Initial Cuts']['EQP Basic'] = time.time() - cut_start_time
        model._nums['Initial Cuts']['EQP Basic'] = len(eqp_cuts)

    def gen_CompleteSolution(self):

        def CompleteSolution(model, data, soln, where):

            if where not in ['Warm Start', 'Callback']:
                print(f'Complete solution requested from EQP Basic but where argument is not recognised')
                return

            try:
                eqp_cuts = data['EQP Cuts']
            except:
                print(f'Complete solution requested from EQP Basic by {where} but EQP Cuts are not available in data dictionary')
                return

            max_removed = self.opts['Features Removed']

            # No auxiliary variables are use if EQP cuts are generated from the full feature dataset
            if max_removed < 1:
                return

            F = data['F']
            tree = model._tree

            b = soln['b']

            beta = model._variables['beta']
            betaV = {}

            for num_removed in range(1,max_removed+1):
                for removed_features in itertools.combinations(F, num_removed):
                    if removed_features in beta:
                        if len(removed_features) == 1:
                            betaV[removed_features] = min(1, sum(b[n,removed_features[0]] for n in tree.B))
                        else:
                            betaV[removed_features] = min(1, sum(betaV[(f,)] for f in removed_features))

            if where == 'Warm Start':
                for k,v in betaV.items():
                    beta[k].Start = v

            elif where == 'Callback':
                model.cbSetSolution(beta, betaV)

        return CompleteSolution

class EQPChain(InitialCut):
    def __init__(self):
        self.cut_name = 'EQP Chain'
        self.default_settings = {'Enabled': False,
                                 'Features Removed': 1,
                                 'Disaggregate Alpha': False,
                                 'Lazy': 0}
        self.opts = self.default_settings

    def validate_settings(self, opts, model_opts=None, instance=None):
        features_removed = opts['Features Removed']

        # Check if new settings are valid for the new opts
        if features_removed not in [0, 1, 2,3]:
            return False

        return True

    def useful_settings(self, opts, model_opts=None, instance=None):
        features_removed = opts['Features Removed']

        if instance is not None:
            eqp_datasets = valid_datasets[f'eqp{features_removed}']
            if instance in valid_datasets['categorical']:
                eqp_encoded_datasets = eqp_datasets['categorical']
            elif instance in valid_datasets['numerical']:
                encoding_scheme = model_opts['Encoding Scheme'] + '-' + str(model_opts['Number Buckets'])
                eqp_encoded_datasets = eqp_datasets['numerical'][encoding_scheme]
            elif instance in valid_datasets['mixed']:
                encoding_scheme = model_opts['Encoding Scheme'] + '-' + str(model_opts['Number Buckets'])
                eqp_encoded_datasets = eqp_datasets['mixed'][encoding_scheme]

        if instance not in eqp_encoded_datasets:
            return False

        return True

    def add_cuts(self,model,data):
        F = data['F']
        X = data['X']
        y = data['y']

        tree = model._tree

        b = model._variables['b']
        theta = model._variables['theta']

        cut_start_time = time.time()

        max_removed = self.opts['Features Removed']
        disagg_alpha = self.opts['Disaggregate Alpha']
        Lazy = self.opts['Lazy']

        eqp_cuts = find_split_sets(data, max_removed=max_removed)
        data['EQP Cuts'] = eqp_cuts

        alpha = {}
        chain_cons = {}

        for cut_idx, rhs_bound, removed_features in eqp_cuts:
            F_star = removed_features
            if len(removed_features) == 0:
                rhs = rhs_bound
            else:
                i = cut_idx[0]
                F_support = [f for f in F if f not in F_star]
                for n in tree.B:
                    alpha_vtype = GRB.CONTINUOUS if disagg_alpha else GRB.BINARY
                    alpha[(F_star,cut_idx,n)] = model.addVar(vtype=alpha_vtype)

                    path = tree.ancestors(n, branch_dirs=True)
                    path_left, path_right = [], []
                    for n_a, d in path.items():
                        if d == 0:
                            path_left.append(n_a)
                        elif d == 1:
                            path_right.append(n_a)
                        else:
                            raise Exception('???')
                    if disagg_alpha:
                        chain_cons[(F_star,cut_idx,n,'split')] = model.addConstr(alpha[(F_star,cut_idx,n)] <= quicksum(b[n,f] for f in F_star))
                        for n_a in path_left:
                            chain_cons[(F_star,cut_idx,n,'path left')] = model.addConstr(alpha[(F_star,cut_idx,n)] <=
                                                                                         quicksum(b[n_a,f] for f in F_support if X[i,f] == 0))
                        for n_a in path_right:
                            chain_cons[(F_star,cut_idx,n,'path right')] = model.addConstr(alpha[(F_star,cut_idx, n)] <=
                                                                                          quicksum(b[n_a, f] for f in F_support if X[i, f] == 1))
                    else:
                        coeff = 1 / (1 + len(path))
                        rhs_sum = (quicksum(b[n,f] for f in F_star) +
                                   quicksum(b[n_a,f] for n_a in path_left for f in F_support if X[i,f] == 0) +
                                   quicksum(b[n_a,f] for n_a in path_right for f in F_support if X[i,f] == 1))

                        chain_cons[(F_star,cut_idx,n,'aggregated')] = model.addConstr(alpha[(F_star,cut_idx,n)] <= coeff * rhs_sum)

                rhs = rhs_bound + (len(cut_idx) - rhs_bound) * quicksum(alpha[(F_star,cut_idx,n)] for n in tree.B)

            chain_cons[(F_star,cut_idx,'cut')] = model.addConstr(quicksum(theta[i] for i in cut_idx) <= rhs)

        if self.opts['Lazy'] != 0:
            for k in chain_cons:
                chain_cons[k].Lazy = Lazy

        model._variables['alpha'] = alpha

        model._times['Initial Cuts']['EQP Chain'] = time.time() - cut_start_time
        model._nums['Initial Cuts']['EQP Chain'] = len(eqp_cuts)

    def gen_CompleteSolution(self):

        def CompleteSolution(model, data, soln, where):

            if where not in ['Warm Start', 'Callback']:
                print(f'Complete solution requested from EQP Chain but where argument is not recognised')
                return

            try:
                eqp_cuts = data['EQP Cuts']
            except:
                print(f'Complete solution requested from EQP Chain by {where} but EQP Cuts are not available in data dictionary')
                return

            max_removed = self.opts['Features Removed']

            # No auxiliary variables are use if EQP cuts are generated from the full feature dataset
            if max_removed < 1:
                return

            X = data['X']
            F = data['F']
            tree = model._tree

            b = soln['b']

            alpha = model._variables['alpha']
            alphaV = {}

            branch_features = {}
            for n in tree.B:
                for f in F:
                    if b[n,f] > 0.9:
                        branch_features[n] = f
                        break

            for cut_idx, rhs_bound, removed_features in eqp_cuts:
                if len(removed_features) == 0:
                    continue

                i = cut_idx[0]
                F_star = removed_features

                for n in tree.B:
                    alphaV[F_star, cut_idx, n] = 0

                node = 1
                while node in tree.B:
                    bf = branch_features[node]
                    if bf in F_star:
                        alphaV[F_star, cut_idx, node] = 1
                        break

                    if X[i,branch_features[node]] > 0.5:
                        node = tree.right_child(node)
                    else:
                        node = tree.left_child(node)

            if where == 'Warm Start':
                for k,v in alphaV.items():
                    alpha[k].Start = v

            elif where == 'Callback':
                model.cbSetSolution(alpha, alphaV)

        return CompleteSolution

class EQPChainAlt1(InitialCut):
    def __init__(self):
        self.cut_name = 'EQP Chain Alt1'
        self.default_settings = {'Enabled': False,
                                 'Features Removed': 1,
                                 'Disaggregate': False,
                                 'Aggregate Cuts': False}
        self.opts = self.default_settings

    def useful_settings(self, opts, model_opts=None, instance=None):
        features_removed = opts['Features Removed']

        if instance is not None:
            eqp_datasets = valid_datasets[f'eqp{features_removed}']
            if instance in valid_datasets['categorical']:
                eqp_encoded_datasets = eqp_datasets['categorical']
            elif instance in valid_datasets['numerical']:
                encoding_scheme = model_opts['Encoding Scheme'] + '-' + str(model_opts['Number Buckets'])
                eqp_encoded_datasets = eqp_datasets['numerical'][encoding_scheme]
            elif instance in valid_datasets['mixed']:
                encoding_scheme = model_opts['Encoding Scheme'] + '-' + str(model_opts['Number Buckets'])
                eqp_encoded_datasets = eqp_datasets['mixed'][encoding_scheme]

        if instance not in eqp_encoded_datasets:
            return False

        return True

    def add_cuts(self,model,data):
        F = data['F']
        X = data['X']
        y = data['y']

        tree = model._tree

        b = model._variables['b']
        theta = model._variables['theta']

        cut_start_time = time.time()

        max_removed = self.opts['Features Removed']
        disagg = self.opts['Disaggregate']

        eqp_cuts = find_split_sets(data, max_removed=max_removed)
        data['EQP Cuts'] = eqp_cuts

        alpha = {}

        for cut_idx, rhs_bound, removed_features in eqp_cuts:
            if len(removed_features) == 0:
                rhs = rhs_bound
            else:
                i = cut_idx[0]
                F_star = removed_features
                F_support = [f for f in F if f not in F_star]

                for n in reversed(tree.B):
                    alpha[cut_idx, n] = model.addVar(vtype=GRB.CONTINUOUS)
                    if n in tree.layers[-2]:
                        model.addConstr(alpha[cut_idx,n] <= quicksum(b[n,f] for f in F_star))
                    else:
                        alpha_vtype = GRB.CONTINUOUS if disagg else GRB.BINARY
                        alpha[cut_idx, n, 'r'] = model.addVar(vtype=alpha_vtype)
                        alpha[cut_idx, n, 'l'] = model.addVar(vtype=alpha_vtype)

                        if disagg:
                            model.addConstr(alpha[cut_idx, n, 'r'] <= quicksum(b[n,f] for f in F_support if X[i,f] == 1))
                            model.addConstr(alpha[cut_idx, n, 'r'] <= alpha[cut_idx,tree.right_child(n)])
                            model.addConstr(alpha[cut_idx, n, 'l'] <= quicksum(b[n, f] for f in F_support if X[i, f] == 0))
                            model.addConstr(alpha[cut_idx, n, 'l'] <= alpha[cut_idx, tree.left_child(n)])
                        else:
                            model.addConstr(2 * alpha[cut_idx, n, 'r'] <=
                                            quicksum(b[n,f] for f in F_support if X[i,f] == 1) + alpha[cut_idx,tree.right_child(n)])
                            model.addConstr(2 * alpha[cut_idx, n, 'l'] <=
                                            quicksum(b[n, f] for f in F_support if X[i,f] == 0) + alpha[cut_idx, tree.left_child(n)])

                        model.addConstr(alpha[cut_idx,n] <= quicksum(b[n,f] for f in F_star) + alpha[cut_idx, n, 'r'] + alpha[cut_idx, n, 'l'])
                rhs = rhs_bound + (len(cut_idx) - rhs_bound) * alpha[(cut_idx, 1)]

            model.addConstr(quicksum(theta[i] for i in cut_idx) <= rhs)

        model._variables['alpha'] = alpha

        model._times['Initial Cuts']['EQP Chain Alt1'] = time.time() - cut_start_time
        model._nums['Initial Cuts']['EQP Chain Alt1'] = len(eqp_cuts)

class EQPChainAlt2(InitialCut):
    def __init__(self):
        self.cut_name = 'EQP Chain Alt2'
        self.default_settings = {'Enabled': False,
                                 'Features Removed': 1,
                                 'Recursive': False,
                                 'Disaggregate': False}
        self.opts = self.default_settings

    def validate_settings(self, opts, model_opts=None, instance=None):
        features_removed = opts['Features Removed']

        # Check if new settings are valid for the new opts
        if features_removed not in [0, 1, 2]:
            return False

        return True

    def useful_settings(self, opts, model_opts=None, instance=None):
        features_removed = opts['Features Removed']

        if instance is not None:
            eqp_datasets = valid_datasets[f'eqp{features_removed}']
            if instance in valid_datasets['categorical']:
                eqp_encoded_datasets = eqp_datasets['categorical']
            elif instance in valid_datasets['numerical']:
                encoding_scheme = model_opts['Encoding Scheme'] + '-' + str(model_opts['Number Buckets'])
                eqp_encoded_datasets = eqp_datasets['numerical'][encoding_scheme]
            elif instance in valid_datasets['mixed']:
                encoding_scheme = model_opts['Encoding Scheme'] + '-' + str(model_opts['Number Buckets'])
                eqp_encoded_datasets = eqp_datasets['mixed'][encoding_scheme]

        if instance not in eqp_encoded_datasets:
            return False

        return True

    def add_cuts(self,model,data):
        F = data['F']
        X = data['X']
        y = data['y']

        tree = model._tree

        b = model._variables['b']
        theta = model._variables['theta']

        cut_start_time = time.time()

        max_removed = self.opts['Features Removed']
        recursive_constraints = self.opts['Recursive']
        disagg = self.opts['Disaggregate']

        eqp_cuts = Equivalent_Points(X, y, max_removed=max_removed)
        data['EQP Cuts'] = eqp_cuts

        alpha = {}

        for cut_idx, rhs_bound, removed_features in eqp_cuts:
            if len(removed_features) == 0:
                if disagg:
                    rhs = [rhs_bound]
                else:
                    rhs = rhs_bound
            else:
                i = cut_idx[0]
                F_support = [f for f in F if f not in removed_features]

                if recursive_constraints:
                    for n in tree.B + tree.L:
                        alpha[cut_idx,n] = model.addVar(vtype=GRB.CONTINUOUS,lb=0)
                    for n in tree.B:
                        if n == 1:
                            model.addConstr(alpha[cut_idx,n] == 1)
                        model.addConstr(alpha[cut_idx,tree.left_child(n)] >= alpha[cut_idx,n] + quicksum(b[n,f] for f in F_support if X[i,f] == 0) - 1)
                        model.addConstr(alpha[cut_idx, tree.right_child(n)] >= alpha[cut_idx, n] + quicksum(b[n, f] for f in F_support if X[i, f] == 1) - 1)


                else:
                    for n in tree.L:
                        alpha[cut_idx,n] = model.addVar(vtype=GRB.CONTINUOUS, lb=0)

                        path = tree.ancestors(n, branch_dirs=True)
                        path_left, path_right = [], []
                        for n_a, d in path.items():
                            if d == 0:
                                path_left.append(n_a)
                            elif d == 1:
                                path_right.append(n_a)
                            else:
                                raise Exception('???')

                        model.addConstr(alpha[cut_idx, n] >= (1 - len(path)
                                                              + quicksum(b[n_a,f] for n_a in path_left for f in F_support if X[i,f] == 0)
                                                              + quicksum(b[n_a,f] for n_a in path_right for f in F_support if X[i,f] == 1)))

                if disagg:
                    rhs = [(len(cut_idx) - (len(cut_idx) - rhs_bound) * alpha[cut_idx,n]) for n in tree.L]
                else:
                    rhs = len(cut_idx) - (len(cut_idx) - rhs_bound) * quicksum(alpha[cut_idx,n] for n in tree.L)


            if disagg:
                for rhs_n in rhs:
                    model.addConstr(quicksum(theta[i] for i in cut_idx) <= rhs_n)
            else:
                model.addConstr(quicksum(theta[i] for i in cut_idx) <= rhs)

        model._variables['alpha'] = alpha

        model._times['Initial Cuts']['EQP Chain Alt2'] = time.time() - cut_start_time
        model._nums['Initial Cuts']['EQP Chain Alt2'] = len(eqp_cuts)

class EQPTarget(InitialCut):
    def __init__(self):
        self.cut_name = 'EQP Target'
        self.default_settings = {'Enabled': False,
                                 'Top Percentage': 0.5}
        self.opts = self.default_settings

    def add_cuts(self,model,data):
        I = data['I']
        F = data['F']
        X = data['X']
        y = data['y']

        tree = model._tree

        b = model._variables['b']
        theta = model._variables['theta']

        cut_start_time = time.time()

        max_removed = self.opts['Max Features Removed']

        support_sets = find_split_sets(X,y,compress_if_same_split_set=True,max_features_removed=max_removed)

        if max_removed >= 1:
            # For practical reasons beta is indexed by tuples which allows for generalisation to betas with multiple features
            beta = {(f,): model.addVar(vtype=GRB.CONTINUOUS, name=f'beta_{f}')
                    for f in F}
            link_betas = {f: model.addConstr(beta[(f,)] <= quicksum(b[n, f] for n in tree.B), name=f'link_beta_{f}')
                          for f in F}
            model._variables['beta'] = beta

        for Fs, Fs_dict in support_sets.items():
            F_star = tuple([f for f in F if f not in Fs])

            cut_idx = Fs_dict['Samples']
            rhs_bound = Fs_dict['Num']

            if len(F_star) == 0:
                rhs = rhs_bound
            else:
                # Cover case when multiple features removed
                if F_star not in beta:
                    beta[F_star] = model.addVar(vtype=GRB.CONTINUOUS)
                    model.addConstr(beta[F_star] <= quicksum(beta[(f,)] for f in F_star))

                rhs = rhs_bound + (len(cut_idx) - rhs_bound) * beta[F_star]

            model.addConstr(quicksum(theta[i] for i in cut_idx) <= rhs)

        model._times['Initial Cuts']['EQP Target'] = time.time() - cut_start_time
        model._nums['Initial Cuts']['EQP Target'] = len(support_sets)

    # def add_cuts(self,model,data):
    #     assert (not data['compressed'])
    #
    #     tree = model._tree
    #
    #     b = model._variables['b']
    #     w = model._variables['w']
    #     theta = model._variables['theta']
    #
    #     X = data['X']
    #     y = data['y']
    #     I = data['I']
    #     F = data['F']
    #
    #     EPS = 1e-4
    #
    #     max_features_removed = self.opts['Max Features Removed']
    #
    #     support_sets = find_split_sets(X,y)
    #     support_set_idx = [(idx, len(F) - len(Fs)) for idx, Fs in support_sets.items()]
    #     support_set_idx.sort(key=lambda x: x[1])
    #
    #     support_set_cut_idx = set(z[0] for z in support_set_idx[:int(percent_kept * len(support_set_idx))])
    #     # support_set_cut_idx = set(support_sets.keys())
    #
    #     # Relax the MP
    #     for k in b:
    #         b[k].vtype = GRB.CONTINUOUS
    #
    #     cut_start_time = time.time()
    #     LogToConsoleSetting = model.Params.LogToConsole
    #     LogFile = model.Params.LogFile
    #     model.Params.LogFile = ''
    #     model.Params.LogToConsole = 0
    #
    #     if 'beta' in model._variables:
    #         beta = model._variables['beta']
    #     else:
    #
    #         beta = {f: model.addVar(vtype=GRB.CONTINUOUS)
    #                 for f in F}
    #         link_betas = {f: model.addConstr(beta[f] <= quicksum(b[n, f] for n in tree.B), name=f'link_beta_{f}')
    #                       for f in F}
    #         model._variables['beta'] = beta
    #
    #     num_cuts_added = 0
    #
    #     iteration = 1
    #
    #     while True:
    #         cuts_added = 0
    #         model_opt_start_time = time.time()
    #         model.optimize()
    #         model_opt_time = time.time() - model_opt_start_time
    #         print(model.ObjVal)
    #
    #         # if iteration > 1:
    #         #     break
    #
    #         updated_support_set_cut_idx = set()
    #
    #         cut_gen_start_time = time.time()
    #
    #         for cut_idx in support_set_cut_idx:
    #             F_support = support_sets[cut_idx]
    #             F_star = tuple(f for f in F if f not in F_support)
    #
    #             if len(F_star) == 0:
    #                 continue
    #
    #             if (sum(theta[i].X for n in tree.L for i in cut_idx) > 1 + EPS) and (sum(b[n,f].X for n in tree.B for f in F_star) < 0.01):
    #                 rhs = 1 + quicksum(beta[f] for f in F_star)
    #                 model.addConstr(quicksum(theta[i] for i in cut_idx) <= rhs)
    #
    #                 cuts_added += 1
    #             else:
    #                 updated_support_set_cut_idx.add(cut_idx)
    #
    #         support_set_cut_idx = updated_support_set_cut_idx
    #         num_cuts_added += cuts_added
    #
    #         cut_gen_time = time.time() - cut_gen_start_time
    #
    #         print(f'Iteration {iteration}: Spent {model_opt_time:.1f}s optimising the LP and {cut_gen_time:.1f}s adding {cuts_added} cuts')
    #         iteration += 1
    #
    #
    #
    #         if cuts_added == 0:
    #             break
    #
    #     print(f'Added {num_cuts_added} cuts from {len(support_sets)} possible cuts')
    #
    #     # Unrelax the MP
    #     for k in b:
    #         b[k].vtype = GRB.BINARY
    #
    #
    #     model.Params.LogToConsole = LogToConsoleSetting
    #     model.Params.LogFile = LogFile

def BendOCTWrapper(hyperparameters,
                   datasets,
                   opt_params={},
                   gurobi_params={}):

    cuts = [SubproblemLP(), SubproblemDualInspection(), EQPBasic(),
            EQPChain(), EQPTarget(), NoFeatureReuse(),
            EQPChainAlt1(), EQPChainAlt2()]
    callback_generator = BendersCallback()

    available_cuts = {cut_object.cut_name: cut_object
                      for cut_object in cuts}

    # These are the default parameters
    opt_params_default = {'Warmstart': True,
                          'Polish Warmstart': True,
                          'Base Directory': os.getcwd(),
                          'Initial Cuts': {cut_name: cut.default_settings for cut_name, cut in available_cuts.items()},
                          'Callback': callback_generator.default_settings,
                          'Compress Data': False,
                          'Results Directory': 'Test Folder',
                          'depth': 3,
                          'Use Baseline': False,
                          'Encoding Scheme': None,
                          'Number Buckets': None}

    gurobi_params_default = {'TimeLimit': 3600,
                             'Threads': 1,
                             'MIPGap': 0,
                             'MIPFocus': 0,
                             'Heuristics': 0.05,
                             'NodeMethod': -1,
                             'Method': -1,
                             'Seed': 0,
                             'LogToConsole': 0,
                             'LogToFile': True}

    # Update default parameters
    opt_params = opt_params_default | opt_params
    gurobi_params = gurobi_params_default | gurobi_params

    if opt_params['Use Baseline']:
        print('-' * 10 + 'USING BASELINE SETTINGS' + '-'*10)

    # Update default initial cut settings
    for cut_name, cut in available_cuts.items():
        if cut_name in opt_params['Initial Cuts'] or not opt_params['Use Baseline']:
            opt_params['Initial Cuts'][cut_name] = opt_params_default['Initial Cuts'][cut_name] | opt_params['Initial Cuts'][cut_name]
        else:
            opt_params['Initial Cuts'][cut_name] = opt_params_default['Initial Cuts'][cut_name]

    # Update callback settings:
    if 'Callback' in opt_params or not opt_params['Use Baseline']:
        opt_params['Callback'] = opt_params_default['Callback'] | opt_params['Callback']
    else:
        opt_params['Callback'] = opt_params_default['Callback']

    # Set up directories
    base_dir = opt_params.get('Base Directory', os.getcwd())
    results_dir_name = opt_params.get('Results Directory', 'Test Folder')
    results_base_dir = os.path.join(base_dir, 'Results', 'BendOCT', results_dir_name)

    os.makedirs(results_base_dir,exist_ok=True)

    console = sys.stdout
    my_logger = logger(console=console)

    # Retrieve required params
    compress_dataset = opt_params['Compress Data']

    for dataset in datasets:
        results_dir = os.path.join(results_base_dir,dataset)
        os.makedirs(results_dir,exist_ok=True)

        LogFile = os.path.join(results_dir, 'Logfile.txt')
        my_logger.SetLogFile(LogFile)
        sys.stdout = my_logger

        if 'Encoding Scheme' not in hyperparameters:
            ############### Load in the data ###############
            instance_data = load_instance(dataset,
                                          compress=compress_dataset,
                                          encoding_scheme=opt_params['Encoding Scheme'],
                                          num_buckets=opt_params['Number Buckets'])

            # If dataset failed to load then continue to next dataset
            if instance_data is None:
                continue

        ############### Loop over Hyperparameter Combinations ###############
        hp_names, hp_values = hyperparameters.keys(), hyperparameters.values()

        for hp_combo in itertools.product(*hp_values):

            hp = {hp_name: hp_value for hp_name, hp_value in zip(hp_names, hp_combo)}
            hp_combo_string = ','.join(f'{hp_name} = {hp_value}' for hp_name, hp_value in hp.items())

            hp_items_abbrev = []
            for hp_name, hp_value in hp.items():
                split_name = hp_name.split('-')
                if len(split_name) == 2:
                    abbrev_name = name_dict.get(split_name[0], split_name[0]) + '-' + name_dict.get(split_name[1], split_name[1])
                    new_item = (abbrev_name, name_dict.get(hp_value,hp_value))
                else:
                    new_item = (name_dict.get(hp_name, hp_name), name_dict.get(hp_value,hp_value))
                hp_items_abbrev.append(new_item)
            hp_combo_string_abbrev = ','.join(f'{hp_name} = {hp_value}' for hp_name, hp_value in hp_items_abbrev)

            # TODO: Explicitly check if parameters are valid
            # Update the parameter dictionaries with new hyperparameters
            update_hyperparameters(opt_params,hp)
            update_hyperparameters(gurobi_params,hp)
            update_initial_cuts(opt_params['Initial Cuts'],hp)

            # If the encoding scheme is a hyperparameter we may need to reload the dataset
            if 'Encoding Scheme' in hyperparameters:
                instance_data = load_instance(dataset,
                                              compress=compress_dataset,
                                              encoding_scheme=opt_params['Encoding Scheme'],
                                              num_buckets=opt_params['Number Buckets'])

                # If dataset failed to load then continue to next dataset
                if instance_data is None:
                    continue

            ############### Set up and run the model ###############
            print('\n' + '#' * 5 + f' Starting Gurobi BendOCT Solve with ' + hp_combo_string + ' ' + '#' * 5)

            # Update the cut settings with new hyperparameters and verify that they are valid for this dataset
            valid_cut_settings = True
            some_useful_features = False
            for cut, cut_opts in opt_params['Initial Cuts'].items():
                settings_valid, settings_useful = available_cuts[cut].UpdateSettings(cut_opts, model_opts=opt_params, instance=dataset)
                if not settings_valid:
                    print(f'{cut} initial cut settings invalid or cannot be used with {dataset} dataset')
                    print(cut_opts)
                    valid_cut_settings = False
                    continue

                if settings_useful:
                    some_useful_features = True

            if not valid_cut_settings:
                continue


            settings_valid, settings_useful = callback_generator.UpdateSettings(opt_params['Callback'], model_opts=opt_params, instance=dataset)
            if not settings_valid:
                print(f'{cut} Callback settings invalid or cannot be used with {dataset} dataset')
                print(opt_params['Callback'])

            if settings_useful:
                some_useful_features = True

            if (not opt_params['Use Baseline']) and (not some_useful_features):
                print('Optimisation settings do not provide any benefit over the baseline')
                continue

            model = BendOCT(opt_params, gurobi_params)
            model.SetInitialCuts(available_cuts)

            if gurobi_params['LogToFile']:
                GurobiLogFile = os.path.join(results_dir,hp_combo_string_abbrev + ' Gurubi Logs.txt')
                model.SetGurobiLogFile(GurobiLogFile)

            model.build_model(instance_data)
            model.SetCallback(callback_generator)
            model.optimize_model()

            ############### Parse output and save results ###############
            csv_file = os.path.join(results_base_dir,results_dir_name + '.csv')

            logged_results = log_results(model.model)
            save_to_csv(model.model,csv_file,opt_params,gurobi_params,logged_results,dataset,instance_data)

        # Close the logfile in the logger object
        my_logger.CloseLogFile()

def update_hyperparameters(params, hp):
    for p in params:
        if p in hp:
            params[p] = hp[p]

def update_initial_cuts(initial_cuts, hps):
    valid_cut_names = initial_cuts.keys()
    for hp in hps.keys():
        hp_split = hp.split('-')
        if len(hp_split) > 1:
            # Input hyperparameters is incorrect if len(hp_split) > 2
            assert (len(hp_split) <= 2)
            cut_name, setting_name = hp_split

            if cut_name in initial_cuts:
                if setting_name in initial_cuts[cut_name]:
                    initial_cuts[cut_name][setting_name] = hps[hp]
                else:
                    print(f'Cannot update settings with hyperparameter {hp} since {setting_name} is not a known setting for {cut_name} initial cuts')
            else:
                print(f'Cannot update settings with hyperparameter {hp} since {cut_name} is not a known initial cut type')


def log_results(model):
    # Check status code on model to determine if solve was successful
    status = model.Status

    # TODO: Maybe update to match...case? But might break on python versions below 3.10
    # TODO: Handle infeasible & unbounded models gracefully. Maybe return {'status': status} and write N/A is all other entries
    if status == GRB.INFEASIBLE:
        print('Model Infeasible'); return
    if status == GRB.UNBOUNDED:
        print('Model Unbounded'); return
    if status == GRB.NUMERIC:
        print('Gurobi terminated solve due to numerical issues'); return

    print('\n' + '-'*10)
    # If successful continue with saving the model output
    if status == 2:
        print(f'Gurobi found optimal solution with objective {model.ObjVal}')
    else:
        if model.ModelSense == 1:
            print(f'Gurobi found solution with objective {model.ObjVal}, lower bound {model.ObjBound:.2f}, and gap {100 * model.MIPGap:.1f}%')
        elif model.ModelSense == -1:
            print(f'Gurobi found solution with objective {model.ObjVal}, upper bound {model.ObjBound:.2f}, and gap {100 * model.MIPGap:.1f}%')
        else:
            raise Exception('Model does not have a sense??')

    print(f'Total Solve Time: {model.runTime:.1f}s' + (' (TIME LIMIT)' if (status == GRB.TIME_LIMIT) else ''))

    if len(model._times['Heuristics']) > 0:
        print(f'\nTime in Initial Heuristics:')
        for heuristic_type in model._times['Heuristics'].keys():
            try:
                heur_time = model._times['Heuristics'][heuristic_type]
                heur_obj = model._nums['Heuristics'][heuristic_type]
                print(f'{heuristic_type} - Obj = {heur_obj} in {heur_time:.1f}s')
            except KeyError:
                print(f'ERROR: MISMATCH IN {heuristic_type} KEY BETWEEN model._times AND model._nums')

    if len(model._times['Initial Cuts']) > 0:
        print(f'\nTime in Initial Cuts:')
        for cut_type in model._times['Initial Cuts'].keys():
            try:
                cut_time = model._times['Initial Cuts'][cut_type]
                num_cuts = model._nums['Initial Cuts'][cut_type]
                print(f'{cut_type} - Produced {num_cuts} cuts in {cut_time:.1f}s')
            except KeyError:
                print(f'ERROR: MISMATCH IN {cut_type} KEY BETWEEN model._times AND model._nums')

    if len(model._times['Callback']) > 0:
        print(f'\nTime in Callback Cuts:')
        for cut_type in model._times['Callback'].keys():
            try:
                cut_time = model._times['Callback'][cut_type]
                num_cuts = model._nums['Callback'][cut_type]
                print(f'{cut_type} - Produced {num_cuts} cuts in {cut_time:.1f}s')
            except KeyError:
                print(f'ERROR: MISMATCH IN {cut_type} KEY BETWEEN model._times AND model._nums')

    print('-' * 10 + '\n')

    results_for_csv = {'Model Status': status,
                       'Objective': model.ObjVal,
                       'Bound': model.ObjBound,
                       'Gap': model.MIPGap,
                       'Solve Time': model.Runtime,
                       'Node Count': model.NodeCount}

    return results_for_csv

def save_to_csv(model,csv_file,opt_params,gurobi_params,saved_columns,dataset,data):
    # This function relies on the unpacking order of dictionaries being stable
    # provided that no entries as added or deleted (value changes are allowed)

    new_file = not os.path.exists(csv_file)

    F = data['F']
    K = data['K']
    cat2bin = data['Categorical Feature Map']
    num2bin = data['Numerical Feature Map']


    columns = ['Model', 'Dataset', '|F|', '|F^C|', '|F^N|', '|F^B|', '|K|']
    row = ['BendOCT', dataset, len(cat2bin) + len(num2bin), len(num2bin), len(cat2bin), len(F), len(K)]

    for c, r in saved_columns.items():
        columns.append(c)
        row.append(r)

    for param_name, param_value in gurobi_params.items():
        columns.append(param_name)
        row.append(param_value)

    for param_name, param_value in opt_params.items():
        if param_name not in ['Initial Cuts', 'Base Directory', 'Results Directory', 'Callback']:
            columns.append(param_name)
            row.append(param_value)

    for cut_type, cut_opts in opt_params['Initial Cuts'].items():
        cut_enabled = cut_opts['Enabled']
        columns.append(f'{cut_type}-Enabled')
        row.append(cut_enabled)
        if cut_enabled:
            for opt_name, opt_value in cut_opts.items():
                if opt_name == 'Enabled':
                    continue
                else:
                    columns.append(f'{cut_type}-{opt_name}')
                    row.append(opt_value)

    callback_opts = opt_params['Callback']
    columns.append(f'Callback-Enabled')
    row.append(callback_opts['Enabled'])
    if callback_opts['Enabled']:
        for opt_name, opt_value in callback_opts.items():
            if opt_name == 'Enabled':
                continue
            else:
                columns.append(f'Callback-{opt_name}')
                row.append(opt_value)

    try:
        with open(csv_file, 'a', newline='') as f:
            csv_writer = csv.writer(f)

            if new_file:
                csv_writer.writerow(columns)
            csv_writer.writerow(row)
    except:
        print('Could not access .csv to write to')

