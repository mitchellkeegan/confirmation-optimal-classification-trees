import os
import sys
import itertools
import time
import csv

from OptimisationModels import OCT, InitialCut, GenCallback
from Utils import CART_Heuristic, Equivalent_Points, logger, find_split_sets, optimise_subtrees, optimise_depth2_subtree
from DataUtils import load_instance, valid_datasets

from gurobipy import *

class FlowOCT(OCT):
    def __init__(self,opt_params, gurobi_params):

        super().__init__(opt_params, gurobi_params)
        self.model_type = 'FlowOCT'

    def add_vars(self,model,data):
        I = data['I']
        F = data['F']
        K = data['K']

        tree = model._tree

        b = {(n, f): model.addVar(vtype=GRB.BINARY, name=f'b_{n}{f}')
                  for n in tree.B for f in F}
        w = {(k, n): model.addVar(vtype=GRB.CONTINUOUS, name=f'w_{k}^{n}')
                  for k in K for n in tree.L}

        # Add flow variables for edges leaving branch nodes
        z = {(n1, n2, i): model.addVar(vtype=GRB.BINARY, name=f'z_{n1}{n2}^{i}')
                  for n1 in tree.B for n2 in tree.children(n1) for i in I}

        # Add in variables for flow from the source node and to the sink node
        for i in I:
            z[(tree.source, 1, i)] = model.addVar(vtype=GRB.BINARY, name=f'z_{tree.source}{1}^{i}')
            for n in tree.L:
                z[(n, tree.sink, i)] = model.addVar(vtype=GRB.BINARY, name=f'z_{n}{tree.sink}^{i}')


        model._variables = {'b': b,
                            'w': w,
                            'z': z}

    def add_constraints(self,model,data):
        I = data['I']
        F = data['F']
        K = data['K']
        X = data['X']
        y = data['y']

        b = model._variables['b']
        w = model._variables['w']
        z = model._variables['z']

        tree = model._tree

        # Can only branch on one variable at each branch node
        only_one_branch = {n: model.addConstr(quicksum(b[n, f] for f in F) == 1, name=f'One Branch Feature Node {n}')
                           for n in tree.B}

        # Flow in = flow out at branch nodes
        branch_flow_equality = {(n, i): model.addConstr(z[tree.parent(n), n, i] ==
                                                        quicksum(z[n, n_child, i] for n_child in tree.children(n)), name=f'Branch Flow Equality Node {n} Sample {i}')
                                for n in tree.B for i in I}

        # Flow in = flow out at leaf nodes
        leaf_flow_equality = {(n, i): model.addConstr(z[tree.parent(n), n, i] == z[n, tree.sink, i], name=f'Leaf Flow Equality Node {n} Sample {i}')
                              for n in tree.L for i in I}

        # Flow from source at most one
        source_flow_bound = {i: model.addConstr(z[tree.source, 1, i] <= 1)
                             for i in I}

        # Bound the left child flow capacity at each branch node
        left_child_capacity = {(n, i): model.addConstr(z[n, tree.left_child(n), i] <=
                                                       quicksum(b[n, f] for f in F if X[i, f] == 0), name=f'Left Child Capacity Node {n} Sample {i}')
                               for n in tree.B for i in I}

        # Bound the right child flow capacity at each branch node
        right_child_capacity = {(n, i): model.addConstr(z[n, tree.right_child(n), i] <=
                                                        quicksum(b[n, f] for f in F if X[i, f] == 1), name=f'Right Child Capacity Node {n} Sample {i}')
                                for n in tree.B for i in I}

        # Set capacity of edges from leaves to sink node
        sink_flow_bound = {(n, i): model.addConstr(z[n, tree.sink, i] <= w[y[i], n], name=f'Leaf to Sink Capacity Node {n} Sample {i}')
                           for n in tree.L for i in I}

        # Make a single class prediction at each leaf node
        leaf_prediction = {n: model.addConstr(quicksum(w[k, n] for k in K) == 1, name=f'One Prediction at Leaf {n}')
                           for n in tree.L}

        cons = None
        model._cons = cons

    def add_objective(self,model,data):
        tree = model._tree

        z = model._variables['z']
        I = data['I']
        weights = data['weights']

        model.setObjective(quicksum(weights[i] * z[(n,tree.sink,i)] for n in tree.L for i in I), GRB.MAXIMIZE)

    def warm_start(self,model,data):

        compressed = data['compressed']
        tree = model._tree

        model._opts.add('CART flow vars')
        model._opts.add('CART polish solutions')

        if compressed:
            X, y = data['Xf'], data['yf']
        else:
            X, y = data['X'], data['y']

        b = model._variables['b']
        w = model._variables['w']
        z = model._variables['z']

        heuristic_start_time = time.time()

        HeuristicSoln = CART_Heuristic(X, y, tree, model._opts, cat_feature_maps=data['Categorical Feature Map'])

        if HeuristicSoln is not None:
            for k, v in HeuristicSoln['b'].items():
                b[k].Start = v

            for k, v in HeuristicSoln['w'].items():
                w[k].Start = v

            for k, v in HeuristicSoln['z'].items():
                if compressed:
                    # Map the idx to the compressed idx
                    idx_map = data['idxf_to_idxc']
                    n1, n2, i = k
                    j = idx_map[i]
                    z[n1,n2,j].Start = v

                else:
                    z[k].Start = v

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

    def save_model_output(self):
        b, w, z = self.vars_to_readable()

        hp_combo_string = ','.join(f'{hp_name} = {hp_value}' for hp_name, hp_value in self.hp.items())
        soln_var_file = os.path.join(self.results_directory,
                                     self.opt_params['instance'],
                                     hp_combo_string + ' Soln Vars.txt')
        with open(soln_var_file,'w') as f:
            f.write('*'*5 + ' BRANCH VARIABLES ' + '*'*5 + '\nnode:feature\n')
            for node,feature in b:
                f.write(f'{node}:{feature}\n')

            f.write('\n' + '*' * 5 + ' PREDICTION VARIABLES ' + '*' * 5 + '\nleaf:predicted class\n')
            for node, pred in w:
                f.write(f'{node}:{pred}\n')

            f.write('\n' + '*' * 5 + ' FLOW VARIABLES ' + '*' * 5 + '\nsample:sample path\n')
            for i, flow_vars in enumerate(z):
                sample_flow = '->'.join([str(n) for n in flow_vars])
                if sample_flow == '':
                    sample_flow = 'misclassified'
                f.write(f'{i}:{sample_flow}\n')

    def vars_to_readable(self):
        # Convert model output to a readable format for saving

        b = self.b
        w = self.w
        z = self.z

        I = self.instance_data['I']
        F = self.instance_data['F']
        K = self.instance_data['K']

        tree = self.tree

        bS = [(n,f) for n in tree.B for f in F if b[n,f].X > 0.5]
        wS = [(n,k) for n in tree.L for k in K if w[k,n].X > 0.5]
        zS = [[n1 for n1 in tree.B + tree.L for n2 in tree.children(n1) if z[n1,n2,i].X > 0.5] for i in I]

        return bS, wS, zS

class FlowCallback(GenCallback):
    def __init__(self):
        self.callback_name = 'Standard Benders Cuts'
        self.default_settings = {'Enabled': True,
                                 'D2Subtrees Primal Heuristic': False,
                                 'EQP Basic Cutting Planes': False,
                                 'Primal Heuristic Cutting Planes': False}
        self.opts = self.default_settings

    def update_model(self,model):

        if self.opts['D2Subtrees Primal Heuristic']:
            model._nums['Callback']['Polish Solutions'] = 0
            model._times['Callback']['Polish Solutions'] = 0.0

            model._PolishSolutionsCache = {}

        if self.opts['EQP Basic Cutting Planes']:
            model.Params.PreCrush = 1

            data = model._data

            # eqp_cuts = Equivalent_Points(data['X'],data['y'],max_removed=0)
            eqp_cuts = find_split_sets(data,max_removed=None)
            model._CallbackEQPCuts = eqp_cuts

            b = model._variables['b']
            tree = model._tree
            F = data['F']

            beta = {(f,): model.addVar(vtype=GRB.CONTINUOUS, ub=1, name=f'beta_{f}')
                    for f in F}

            link_betas = {f: model.addConstr(beta[(f,)] <= quicksum(b[n, f] for n in tree.B), name=f'link_beta_{f}')
                          for f in F}

            model._variables['beta'] = beta

            model._nums['Callback']['EQP Cutting Planes'] = 0
            model._times['Callback']['EQP Cutting Planes'] = 0.0

        if self.opts['Primal Heuristic Cutting Planes']:
            model.Params.PreCrush = 1

            model._nums['Callback']['PH Cutting Planes'] = 0
            model._times['Callback']['PH Cutting Planes'] = 0.0

            model._PHCuttingPlaneCache = {}

    def gen_callback(self):

        if not self.opts['Enabled']:
            return None

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
                if self.opts['D2Subtrees Primal Heuristic'] and tree.depth > 1:
                    if not data['compressed']:

                        X = data['X']
                        y = data['y']
                        F = data['F']
                        I = data['I']
                        weights = data['weights']
                        cat_feature_maps = data['Categorical Feature Map']
                        num_feature_maps = data['Numerical Feature Map']

                        b = model._variables['b']
                        w = model._variables['w']
                        z = model._variables['z']

                        bV = model.cbGetSolution(b)
                        zV = model.cbGetSolution(z)

                        sample_node_path, samples_in_node, node_branch_feature, cut_branch_vars = DFS(1, I, bV, tree, F, X, cut_vars=False)

                        polish_soln_start_time = time.time()

                        b_subtrees, w_subtrees, theta_polished = optimise_subtrees(X, y, samples_in_node, tree, model._opts, node_branch_feature,
                                                                                   cache=model._PolishSolutionsCache,
                                                                                   weights=weights,
                                                                                   cat_feature_maps=cat_feature_maps,
                                                                                   num_feature_maps=num_feature_maps)

                        if b_subtrees is not None:
                            PossObj = sum(theta_polished)
                            CurrObj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                            if PossObj > CurrObj + 0.1:
                                bV |= b_subtrees
                                wV = w_subtrees

                                model.cbSetSolution(b, bV)
                                model.cbSetSolution(w, wV)

                                # for sc in model._solution_completers:
                                #     sc(model, data, {'b': bV}, 'Callback')

                                model.cbUseSolution()


                                model._nums['Callback']['Polish Solutions'] += 1
                                print(f'**** Callback Primal Heuristic improved solution from {CurrObj} to {PossObj} ****')

                        model._times['Callback']['Polish Solutions'] += time.time() - polish_soln_start_time

            if where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                if self.opts['EQP Basic Cutting Planes']:

                    cut_start_time = time.time()

                    b = model._variables['b']
                    z = model._variables['z']
                    beta = model._variables['beta']

                    bR = model.cbGetNodeRel(b)
                    zR = model.cbGetNodeRel(z)
                    betaR = model.cbGetNodeRel(beta)

                    eqp_cuts = model._CallbackEQPCuts

                    for cut_idx, rhs_bound, removed_features in eqp_cuts:
                        if len(removed_features) == 0:
                            if sum(zR[n, tree.sink, i] for n in tree.L for i in cut_idx) > rhs_bound + EPS:
                                tcon = (quicksum(z[n, tree.sink, i] for n in tree.L for i in cut_idx) <= rhs_bound)
                                model.cbCut(tcon)
                                # print('cut added!')
                                model._nums['Callback']['EQP Cutting Planes'] += 1

                        else:
                            if sum(zR[n, tree.sink, i] for n in tree.L for i in cut_idx) > rhs_bound + (len(cut_idx) - rhs_bound) * sum(bR[n,removed_features[0]] for n in tree.B) + EPS:
                            # if sum(zR[n, tree.sink, i] for n in tree.L for i in cut_idx) > rhs_bound + (len(cut_idx) - rhs_bound) * betaR[removed_features] + EPS:

                                tcon = (quicksum(z[n, tree.sink, i] for n in tree.L for i in cut_idx) <= rhs_bound + (len(cut_idx) - rhs_bound) * quicksum(b[n,removed_features[0]] for n in tree.B))
                                # tcon = (quicksum(z[n, tree.sink, i] for n in tree.L for i in cut_idx) <= rhs_bound + (len(cut_idx) - rhs_bound) * beta[removed_features])
                                model.cbCut(tcon)
                                model._nums['Callback']['EQP Cutting Planes'] += 1

                    model._times['Callback']['EQP Cutting Planes'] += time.time() - cut_start_time

                if self.opts['Primal Heuristic Cutting Planes']:

                    cut_start_time = time.time()

                    F = data['F']

                    b = model._variables['b']
                    z = model._variables['z']

                    bR = model.cbGetNodeRel(b)
                    zR = model.cbGetNodeRel(z)

                    integral_paths = []

                    node_info = [(1,tuple())]

                    while len(node_info) > 0:

                        node, path = node_info.pop()
                        # Check if we terminate with an integral path found
                        if tree.height(node) == 2:
                            integral_paths.append(path)
                            continue

                        # Check if we have an integral branch variable
                        branch_var = None
                        for f in F:
                            if bR[node,f] > 1 - EPS:
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

                        # First determine which subset of the dataset follows the integral path
                        X = data['X']
                        y = data['y']
                        I = data['I']
                        F = data['F']
                        weights = data['weights']

                        path_branch_choices = []

                        # TODO: Rewrite with efficient numpy operations
                        I_P = list(I)
                        for node, branch_var, dir in path:
                            I_P = [i for i in I_P if X[i, branch_var] == dir]
                            path_branch_choices.append((node, branch_var))
                        final_node, _, final_dir = path[-1]

                        if len(I_P) == 0:
                            continue

                        if final_dir == 0:
                            subtree_root = tree.left_child(final_node)
                        elif final_dir == 1:
                            subtree_root = tree.right_child(final_node)

                        path_key = frozenset((branch_var, dir) for _, branch_var, dir in path)

                        if path_key in model._PHCuttingPlaneCache:
                            # print('Avoided double computation')
                            theta_idx = model._PHCuttingPlaneCache[path_key]
                        else:
                            # Find optimal depth 2 subtree for subset I_P
                            _, _, theta_idx = optimise_depth2_subtree(X[I_P, :], y[I_P])

                            if theta_idx is None:
                                # subroutine returns None if something went wrong
                                continue

                        # subroutine returns indices of I_P which were correctly classified. Take len to get optimal objective
                        optimal_subtree_obj = len(theta_idx)

                        # Check if we successfully optimised the subtree and if the upper bound found is violated by the relaxation solution
                        if sum(zR[n, tree.sink, i] for n in tree.L for i in I_P) > optimal_subtree_obj + EPS:
                            # If the cut is violated, calculate the upper bound on the samples
                            # Then apply the cut to the path (and all symmetrical paths)

                            # Get all permutations of branch variables and orderings
                            # Each one corresponds to a possible path through the tree
                            for path_permutation in itertools.permutations(path):

                                # Update the cache with each path permutation that a cut has been added for
                                model._PHCuttingPlaneCache[frozenset((branch_var, dir) for _, branch_var, dir in path_permutation)] = theta_idx

                                node = 1
                                path_branch_choices = []
                                left_nodes = []
                                right_nodes = []
                                left_branch_vars = []
                                right_branch_vars = []

                                for _, branch_var, dir in path_permutation:
                                    path_branch_choices.append((node,branch_var))

                                    if dir == 0:
                                        left_nodes.append(node)
                                        left_branch_vars.append(branch_var)
                                        node = tree.left_child(node)
                                    elif dir == 1:
                                        right_nodes.append(node)
                                        right_branch_vars.append(branch_var)
                                        node = tree.right_child(node)

                                # relaxing_vars_left = quicksum(b[n,f] for n in left_nodes for f in left_branch_vars)
                                # relaxing_vars_right = quicksum(b[n,f] for n in right_nodes for f in right_branch_vars)
                                # relaxing_branch_vars = (len(right_nodes) + len(left_nodes)) - quicksum()

                                relaxing_branch_vars = len(path_branch_choices) - quicksum(b[n,f] for n,f in path_branch_choices)

                                rhs = optimal_subtree_obj + (len(I_P) - optimal_subtree_obj) * relaxing_branch_vars
                                model.cbCut(quicksum(z[n,tree.sink,i] for n in tree.L for i in I_P) <= rhs)

                                model._nums['Callback']['PH Cutting Planes'] += 1

                        else:
                            # Even if we don't add any cuts, update the cache with all permutations of the path
                            for path_permutation in itertools.permutations(path):
                                model._PHCuttingPlaneCache[frozenset((branch_var, dir) for _, branch_var, dir in path_permutation)] = theta_idx
                    model._times['Callback']['PH Cutting Planes'] += time.time() - cut_start_time

        return callback

    def useful_settings(self, opts, model_opts=None, instance=None):
        if opts['Primal Heuristic Cutting Planes'] or opts['EQP Basic Cutting Planes']:
            return True
        else:
            return False

    def validate_settings(self,opts, model_opts=None, instance=None):
        if (opts['Primal Heuristic Cutting Planes'] or opts['D2Subtrees Primal Heuristic']) and instance in valid_datasets['numerical'] + valid_datasets['mixed']:
            print('\nD2Subtrees Primal Heuristic is too slow with numerical features\n')
            return False

        return True

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

        tree = model._tree
        b = model._variables['b']

        cut_start_time = time.time()
        cuts_added = 0

        for Cf in cat_feature_maps:
            if len(Cf) == 1:
                f = Cf[0]
                for n in tree.L:
                    model.addConstr(quicksum(b[n_a,f] for n_a in tree.ancestors(n)) <= 1)
            else:
                for n in tree.B:
                    right_subtree_root = tree.right_child(n)
                    if right_subtree_root in tree.B:
                        subtree, _ = tree.descendants(right_subtree_root, split_nodes=True)

                        cuts_added += 1
                        con_name = f'Root={n}_Group=[' + ','.join(str(f) for f in Cf) + ']'
                        # TODO: Could tighten this but I don't think it's actually very important to do so
                        model.addConstr(quicksum(b[n_d,f] for n_d in subtree for f in Cf) <= len(subtree) * (1 - quicksum(b[n,f] for f in Cf)),
                                        name=con_name)

        model._times['Initial Cuts']['No Feature Reuse'] = 0.0
        model._nums['Initial Cuts']['No Feature Reuse'] = cuts_added

class EQPTarget(InitialCut):
    def __init__(self):
        self.cut_name = 'EQP Target'
        self.default_settings = {'Enabled': False}
        self.opts = self.default_settings

    def add_cuts(self,model,data):
        assert (not data['compressed'])

        tree = model._tree

        b = model._variables['b']
        w = model._variables['w']
        z = model._variables['z']

        X = data['X']
        y = data['y']
        I = data['I']
        F = data['F']

        EPS = 1e-4

        # eqp_cuts = Equivalent_Points(X,y,max_removed=1)

        support_sets = find_split_sets(X,y)
        support_set_cut_idx = set(support_sets.keys())


        # support_sets_by_i = {i: {} for i in I}
        # for IJ, ss in support_sets.items():
        #     i,j = IJ
        #     support_sets_by_i[i][j] = set(ss)

        # Relax the MP
        for k in b:
            b[k].vtype = GRB.CONTINUOUS

        for k in z:
            z[k].vtype = GRB.CONTINUOUS

        cut_start_time = time.time()
        LogToConsoleSetting = model.Params.LogToConsole
        LogFile = model.Params.LogFile
        model.Params.LogFile = ''
        model.Params.LogToConsole = 0

        if 'beta' in model._variables:
            beta = model._variables['beta']
        else:

            beta = {f: model.addVar(vtype=GRB.CONTINUOUS)
                    for f in F}
            link_betas = {f: model.addConstr(beta[f] <= quicksum(b[n, f] for n in tree.B), name=f'link_beta_{f}')
                          for f in F}
            model._variables['beta'] = beta

        num_cuts_added = 0

        while True:
            cuts_added = 0
            model.optimize()
            print(model.ObjVal)

            updated_support_set_cut_idx = set()

            for cut_idx in support_set_cut_idx:
                F_support = support_sets[cut_idx]
                F_star = tuple(f for f in F if f not in F_support)

                if len(F_star) == 0:
                    continue

                if (sum(z[n, tree.sink, i].X for n in tree.L for i in cut_idx) > 1 + EPS) and (sum(b[n,f].X for n in tree.B for f in F_star) < 0.01):
                    rhs = 1 + quicksum(beta[f] for f in F_star)
                    model.addConstr(quicksum(z[n,tree.sink,i] for n in tree.L for i in cut_idx) <= rhs)

                    cuts_added += 1
                else:
                    updated_support_set_cut_idx.add(cut_idx)

            support_set_cut_idx = updated_support_set_cut_idx
            num_cuts_added += cuts_added
            if cuts_added == 0:
                break

        # while True:
        #     cuts_added = 0
        #     model.optimize()
        #     print(model.ObjVal)
        #     # Idea one - Manually enforce
        #     for n in tree.B:
        #         line = f'Node {n} - '
        #         features_used = []
        #         for f in F:
        #             if b[n,f].X > 0.02:
        #                 features_used.append(f'b_{f} = {b[n,f].X:.2f}')
        #         print(line + ','.join(features_used))
        #     print('\n')
        #
        #     for cut_idx, rhs_bound, removed_features in eqp_cuts:
        #         if len(removed_features) == 0:
        #             continue
        #
        #         f_removed = removed_features[0]
        #
        #         if sum(z[n,tree.sink,i].X for n in tree.L for i in cut_idx) > rhs_bound:
        #             # print(f'Cut idx {cut_idx} bound lifted which requires {f_removed} in the tree')
        #             if sum(b[n,f_removed].X for n in tree.B) < 0.01:
        #                 rhs = rhs_bound + (len(cut_idx) - rhs_bound) * beta[f_removed]
        #                 model.addConstr(quicksum(z[n,tree.sink,i] for n in tree.L for i in cut_idx) <= rhs)
        #                 cuts_added += 1
        #
            # num_cuts_added += cuts_added
            # if cuts_added == 0:
            #     break

        print(f'Added {num_cuts_added} cuts from {len(support_sets)} possible cuts')

            # for i in I:

                # # Fill in flow graph capacities
                # cap = {}
                # for n in tree.B:
                #     cap[n, tree.left_child(n)] = sum(b[n, f].X for f in F if X[i, f] == 0)
                #     cap[n, tree.right_child(n)] = sum(b[n, f].X for f in F if X[i, f] == 1)
                # for n in tree.L:
                #     cap[n, tree.sink] = w[y[i], n].X
                #
                # nodes_to_explore = [(1,1)]
                # while len(nodes_to_explore) > 0:
                #
                #     node, flow = nodes_to_explore.pop()

        # Unrelax the MP
        for k in b:
            b[k].vtype = GRB.BINARY

        for k in z:
            z[k].vtype = GRB.BINARY

        model.Params.LogToConsole = LogToConsoleSetting
        model.Params.LogFile = LogFile

class EQPBasic(InitialCut):
    def __init__(self):
        self.cut_name = 'EQP Basic'
        self.default_settings = {'Enabled': False,
                                 'Features Removed': 0}
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
            if instance not in valid_datasets[f'eqp{features_removed}']:
                return False

        return True

    def add_cuts(self,model,data):
        I = data['I']
        F = data['F']
        X = data['X']
        y = data['y']

        tree = model._tree

        b = model._variables['b']
        z = model._variables['z']

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
            con_name = f'{cut_idx}_{rhs_bound}_{removed_features}'
            model.addConstr(quicksum(z[n,tree.sink,i] for n in tree.L for i in cut_idx) <= rhs,
                            con_name)

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

def FlowOCTWrapper(hyperparameters,
                   datasets,
                   opt_params={},
                   gurobi_params={}):

    cuts = [EQPBasic(), NoFeatureReuse(), EQPTarget()]
    callback_generator = FlowCallback()

    available_cuts = {cut_object.cut_name: cut_object
                      for cut_object in cuts}

    # These are the default parameters
    opt_params_default = {'Warmstart': True,
                          'Polish Warmstart': True,
                          'Base Directory': os.getcwd(),
                          'Initial Cuts': {cut: {} for cut in available_cuts},
                          'Callback': callback_generator.default_settings,
                          'Compress Data': False,
                          'Results Directory': 'Test Folder',
                          'depth': 3,
                          'Use Baseline': False,
                          'Encoding Scheme': None}

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
        if cut_name in opt_params['Initial Cuts']:
            opt_params['Initial Cuts'][cut_name] = cut.default_settings | opt_params['Initial Cuts'][cut_name]
        else:
            opt_params['Initial Cuts'][cut_name] = cut.default_settings


    # Set up directories
    base_dir = opt_params.get('Base Directory', os.getcwd())
    results_dir_name = opt_params.get('Results Directory', 'Test Folder')
    results_base_dir = os.path.join(base_dir, 'Results', 'FlowOCT', results_dir_name)

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
                                          encoding_scheme=opt_params['Encoding Scheme'])

            # If dataset failed to load then continue to next dataset
            if instance_data is None:
                continue

        ############### Loop over Hyperparameter Combinations ###############
        hp_names, hp_values = hyperparameters.keys(), hyperparameters.values()

        for hp_combo in itertools.product(*hp_values):

            hp = {hp_name: hp_value for hp_name, hp_value in zip(hp_names, hp_combo)}
            hp_combo_string = ','.join(f'{hp_name} = {hp_value}' for hp_name, hp_value in hp.items())

            # TODO: Explicitly check if parameters are valid
            # Update the paramater dictionaries with new hyperparameters
            update_hyperparameters(opt_params,hp)
            update_hyperparameters(gurobi_params,hp)
            update_initial_cuts(opt_params['Initial Cuts'],hp)

            # If the encoding scheme is a hyperparameter we may need to reload the dataset
            if 'Encoding Scheme' in hyperparameters:
                instance_data = load_instance(dataset,
                                              compress=compress_dataset,
                                              encoding_scheme=opt_params['Encoding Scheme'])

            ############### Set up and run the model ###############
            print('\n' + '#' * 5 + f' Starting Gurobi FlowOCT Solve with ' + hp_combo_string + ' ' + '#' * 5)

            # Update the cut settings with new hyperparameters and verify that they are valid for this dataset
            valid_cut_settings = True
            some_useful_features = False
            for cut, cut_opts in opt_params['Initial Cuts'].items():
                settings_valid, settings_useful = available_cuts[cut].UpdateSettings(cut_opts, model_opts=opt_params,
                                                                                     instance=dataset)
                if not settings_valid:
                    print(f'{cut} initial cut settings invalid or cannot be used with {dataset} dataset')
                    print(cut_opts)
                    valid_cut_settings = False
                    continue

                if settings_useful:
                    some_useful_features = True

            if not valid_cut_settings:
                continue

            settings_valid, settings_useful = callback_generator.UpdateSettings(opt_params['Callback'],
                                                                                model_opts=opt_params, instance=dataset)
            if not settings_valid:
                continue

            if settings_useful:
                some_useful_features = True

            if (not opt_params['Use Baseline']) and (not some_useful_features):
                print('Optimisation settings do not provide any benefit over the baseline')
                continue

            model = FlowOCT(opt_params, gurobi_params)
            model.SetInitialCuts(available_cuts)

            if gurobi_params['LogToFile']:
                GurobiLogFile = os.path.join(results_dir,hp_combo_string + ' Gurubi Logs.txt')
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

    print(f'\nTime in Heuristics:')
    for heuristic_type in model._times['Heuristics'].keys():
        try:
            heur_time = model._times['Heuristics'][heuristic_type]
            heur_obj = model._nums['Heuristics'][heuristic_type]
            print(f'{heuristic_type} - Obj = {heur_obj} in {heur_time:.1f}s')
        except KeyError:
            print(f'ERROR: MISMATCH IN {heuristic_type} KEY BETWEEN model._times AND model._nums')

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
    row = ['FlowOCT', dataset, len(cat2bin) + len(num2bin), len(num2bin), len(cat2bin), len(F), len(K)]


    for c, r in saved_columns.items():
        columns.append(c)
        row.append(r)

    for param_name, param_value in gurobi_params.items():
        columns.append(param_name)
        row.append(param_value)

    for param_name, param_value in opt_params.items():
        if param_name not in ['Initial Cuts', 'Base Directory', 'Results Directory']:
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

