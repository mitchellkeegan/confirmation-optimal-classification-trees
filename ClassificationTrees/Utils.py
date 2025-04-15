import os
import sys
import itertools
import time
import pickle

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

class Node():
    def __init__(self,n,I,F,depth):
        self.n = n
        self.I = I
        self.F = F
        self.depth = depth
        self.left_child, self.right_child = None, None
        self.node_type = None
        self.branch_feature = None
        self.prediction = None

    def make_branch_node(self,f):
        self.node_type = 'branch'
        self.prediction = None

        self.branch_feature = f

    def make_leaf_node(self,y):
        self.node_type = 'prediction'
        self.branch_feature = None
        self.left_child, self.right_child = None, None

        node_y = y[self.I,]

        classes_present, counts = np.unique(node_y, return_counts=True)

        if len(classes_present) == 0:
            self.prediction = 0
            self.num_misclassified = 0
        else:
            self.prediction = classes_present[np.argmax(counts)]
            self.num_misclassified = (node_y != self.prediction).sum()

    def calculate_number_misclassified(self,y):
        # Calculate the number of points misclassified for:
        # a) The leaves in the subtree (the leaf itself if the node is already a leaf)
        # b) The node if it was transformed from a branch node into a leaf node

        # Also calculate the number of leaves
        assert self.node_type is not None

        if self.node_type == 'prediction':
            return self.num_misclassified, 1
        elif self.node_type == 'branch':
            assert self.left_child is not None
            assert self.right_child is not None

            # Calculate the number of sample which would be misclassified if we turned
            # the branch node into a leaf node and cutoff the subtree below
            node_y = y[self.I,]
            classes_present, counts = np.unique(node_y, return_counts=True)

            if len(classes_present) == 0:
                prediction = 0
                self.leaf_num_misclassified = 0
            else:
                prediction = classes_present[np.argmax(counts)]
                self.leaf_num_misclassified = (node_y != prediction).sum()


            left_misclassified, left_num_leaves = self.left_child.calculate_number_misclassified(y)
            right_misclassified, right_num_leaves = self.right_child.calculate_number_misclassified(y)

            # Calculate number of samples misclassified in subtrees recursively
            self.subtree_num_misclassified = left_misclassified + right_misclassified
            self.num_leaves = left_num_leaves + right_num_leaves

            return self.subtree_num_misclassified, self.num_leaves
        else:
            assert False

def cost_complexity_pruning(root,X,y,alpha):
    # Basic idea - calculate effective alpha for all nodes in tree and prune

    num_samples = X.shape[0]

    while True:
        # Calculate the number of misclassified samples
        # Calculating this at the root will also update all other nodes in the tree
        root.calculate_number_misclassified(y)

        to_explore = [root]
        min_alpha = float('inf')
        node_to_prune = None

        # Calculate the effective alpha at each node
        # Prune the subtree with the lowest effective alpha, or stop if all have effective alpha > alpha
        while len(to_explore) > 0:
            node = to_explore.pop()

            if node.node_type == 'prediction':
                continue

            effective_alpha = ((node.leaf_num_misclassified - node.subtree_num_misclassified) / num_samples) / (node.num_leaves - 1)

            if effective_alpha <= min(alpha, min_alpha):
                min_alpha = effective_alpha
                node_to_prune = node

            to_explore.append(node.left_child)
            to_explore.append(node.right_child)

        if node_to_prune is None:
            break

        node_to_prune.make_leaf_node(y)

def node_impurity(y):
    # Takes in a numpy array y and calculates the impurity for a decision tree node with these target classes in it

    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    impurity = np.sum(p * (1-p))

    return impurity

def Custom_CART_Heuristic(X,y,tree,opts,cat_feature_maps=None,num_feature_maps=None,alpha=None):
    # Note: Assumes that the classes of y are from 0,...,|K|-1

    # TODO: Allow for NoFeatureReuse option

    num_samples, num_features = X.shape
    I = range(num_samples)
    F = range(num_features)
    K = np.unique(y).tolist()
    max_depth = tree.depth

    root_node = Node(1,np.asarray(I),np.asarray(F),0)

    # Nodes have the following information
    to_explore = [root_node]

    while len(to_explore) > 0:

        node = to_explore.pop()

        n = node.n
        samples_in_node = node.I
        features_in_node = node.F
        node_depth = node.depth

        # Become a prediction node if we have reached maximum depth
        if node_depth == max_depth:
            node.make_leaf_node(y)

        else:
            node_X = X[samples_in_node, :]
            node_y = y[samples_in_node,]

            node.node_type = 'branch'

            if len(node_y) == 0:
                # If no samples are left in the node, choose an arbitrary branch feature
                # For regularised trees this will be pruned
                # For balanced trees we grow to full size for compatibility with IP model
                best_split_feature = features_in_node[0]
            else:
                best_split_feature = None
                minimum_impurity = float('inf')
                for f in features_in_node:
                    # Get a mask array we can use to index into samples in the left and right children
                    left_child_mask = (node_X[:,f] == 0)

                    # Determine the impurity for the given split feature
                    y_left, y_right = node_y[left_child_mask], node_y[~left_child_mask]
                    impurity_left, impurity_right = node_impurity(y_left), node_impurity(y_right)
                    split_impurity = (len(y_left) * impurity_left + len(y_right) * impurity_right) / len(node_y)

                    if split_impurity < minimum_impurity:
                        best_split_feature = f
                        minimum_impurity = split_impurity

                assert (best_split_feature is not None)

            node.make_branch_node(best_split_feature)

            left_child_mask = (node_X[:,best_split_feature] == 0)
            left_child, right_child = tree.children(n)

            node.left_child = Node(left_child, samples_in_node[left_child_mask], features_in_node, node_depth+1)
            node.right_child = Node(right_child, samples_in_node[~left_child_mask], features_in_node, node_depth + 1)

            to_explore.append(node.left_child)
            to_explore.append(node.right_child)

    if alpha is not None:
        cost_complexity_pruning(root_node, X, y, alpha)

    b = {(n,f): 0 for n in tree.B for f in F}
    p = {n: 0 for n in tree.B}
    w = {(k,n):0 for k in K for n in tree.B + tree.L}
    theta = np.zeros(num_samples)


    branch_feature = {}
    to_explore = [root_node]

    while len(to_explore) > 0:
        node = to_explore.pop()

        n = node.n

        if node.node_type == 'branch':
            b[n,node.branch_feature] = 1
            branch_feature[n] = node.branch_feature
            to_explore.append(node.left_child)
            to_explore.append(node.right_child)
        elif node.node_type == 'prediction':
            p[n] = 1
            w[node.prediction,n] = 1
            correctly_classified_samples = node.I[y[node.I,] == node.prediction]
            theta[correctly_classified_samples,] = 1
        else:
            assert False

    soln_dict = {}

    soln_dict['b'] = b
    soln_dict['w'] = w
    soln_dict['p'] = p
    soln_dict['theta'] = theta

    if 'CART flow vars' in opts:

        # Set all edges to zero
        z = {(n1, n2, i): 0
             for n1 in tree.B for n2 in tree.children(n1) for i in I}

        for i in I:
            z[(tree.source, 1, i)] = 0
            for n in tree.B + tree.L:
                z[(n, tree.sink, i)] = 0

        # Fill in edges which have a flow
        for i in I:
            if theta[i] > 0.5:
                z[(tree.source, 1, i)] = 1
                n = 1
                while True:
                    if p[n] == 1:
                        z[n,tree.sink,i] = 1
                        break

                    else:
                        # Branch node
                        f = branch_feature[n]
                        next_node = tree.right_child(n) if X[i,f] > 0.5 else tree.left_child(n)

                        z[n,next_node,i] = 1
                        n = next_node

        soln_dict['z'] = z

    return soln_dict


    return b,p,w,theta

def CART_Heuristic(X,y,mytree,opts,cat_feature_maps=None,num_feature_maps=None):
    n_samples, n_features = X.shape
    I = range(n_samples)
    F = range(n_features)
    K = np.unique(y).tolist()
    max_depth = mytree.depth

    DecisionTree = DecisionTreeClassifier(max_depth=mytree.depth,ccp_alpha=0.0)
    DecisionTree.fit(X,y)

    children_left = DecisionTree.tree_.children_left
    children_right = DecisionTree.tree_.children_right
    feature = DecisionTree.tree_.feature

    sklearn_to_bfs_map = [0] * DecisionTree.tree_.node_count
    early_leaves = []

    branch_feature = {}
    leaf_prediction = {}
    flow_var_keys = [(mytree.source,1)]
    cut_var_keys = {}

    # Explore the decision tree, mapping CART tree nodes to bfs ordering, and tracking branching and predictions
    stack = [(0,1,0)]
    while len(stack) > 0:
        node, bfs_node, node_depth = stack.pop(0)
        sklearn_to_bfs_map[node] = bfs_node
        left_child = children_left[node]
        right_child = children_right[node]

        # Three cases:
        # 1) Normal branch node
        # 2) Normal leaf node
        # 3) Early leaf node, not at max depth

        # Case 1)
        if left_child != right_child:
            bfs_left_child = mytree.left_child(bfs_node)
            bfs_right_child = mytree.right_child(bfs_node)

            flow_var_keys.append((bfs_node,bfs_left_child))
            flow_var_keys.append((bfs_node,bfs_right_child))

            stack.append((left_child,bfs_left_child,node_depth+1))
            stack.append((right_child,bfs_right_child,node_depth+1))

            branch_feature[bfs_node] = feature[node]

        # Case 2)
        elif (left_child == -1) and (node_depth == max_depth):
            # Get the index of the predicted class in the current lead node and update w
            prediction_idx = np.argmax(DecisionTree.tree_.value[node, 0, :])
            predicted_class = DecisionTree.classes_[prediction_idx]

            leaf_prediction[bfs_node] = predicted_class

        # Case 3)
        elif (left_child == -1) and (node_depth < max_depth):
            prediction_idx = np.argmax(DecisionTree.tree_.value[node, 0, :])
            predicted_class = DecisionTree.classes_[prediction_idx]
            early_leaves.append((bfs_node,node_depth,predicted_class))

        else:
            print('????')

    # If CART did not return a full sized tree, manually fill out the tree by adding branch nodes which branch on feature 0
    # and predicting the class of the early leaves in all descendant leaves
    if len(early_leaves) > 0:
        # print(f'CART TREE FILLING OUT FROM {len(early_leaves)} EARLY LEAVES')
        for n_root,root_depth,root_pred in early_leaves:
            if 'No Feature Reuse' in opts:
                if cat_feature_maps is None and num_feature_maps is None:
                    print('No Feature Reuse option enabled for CART but feature maps not supplied')
                    valid_features = [0] * len(mytree.B)

                else:
                    if 'Threshold Encoding' in opts:
                        thresholded_feature_maps = num_feature_maps
                        onehot_feature_maps = cat_feature_maps
                    else:
                        thresholded_feature_maps = None
                        onehot_feature_maps = cat_feature_maps + num_feature_maps

                    invalid_features = set()
                    bin_to_cat_group = {}
                    bin_to_num_group = {}
                    if onehot_feature_maps is not None:
                        for Cf in onehot_feature_maps:
                            for f in Cf:
                                bin_to_cat_group[f] = set(Cf)
                    if thresholded_feature_maps is not None:
                        for Nf in thresholded_feature_maps:
                            for f in Nf:
                                bin_to_num_group[f] = set(Nf)

                    for n_a, dir in mytree.ancestors(n_root, branch_dirs=True).items():
                        ancestor_branch_feature = branch_feature[n_a]
                        # Ancestor branches right
                        if dir == 1:
                            for f in bin_to_cat_group.get(ancestor_branch_feature,[]):
                                invalid_features.add(f)
                            for f in bin_to_num_group.get(ancestor_branch_feature,[]):
                                if f <= ancestor_branch_feature:
                                    invalid_features.add(f)
                        elif dir == 0:
                            invalid_features.add(ancestor_branch_feature)
                            for f in bin_to_num_group.get(ancestor_branch_feature,[]):
                                if f >= ancestor_branch_feature:
                                    invalid_features.add(f)
                    valid_features = [f for f in F if f not in invalid_features]
            else:
                valid_features = [0] * len(mytree.B)

            stack = [(n_root,root_depth,valid_features)]
            while len(stack) > 0:
                node, node_depth, useable_features = stack.pop(0)

                # Turn into branch node
                if node_depth < max_depth:
                    bf = useable_features[0]
                    branch_feature[node] = bf
                    if 'No Feature Reuse' in opts:
                        if bf in bin_to_cat_group:
                            unuseable_left = [bf]
                            unuseable_right = bin_to_cat_group[bf]
                        elif bf in bin_to_num_group:
                            unuseable_left = [f for f in bin_to_num_group[bf] if f >= bf]
                            unuseable_right = [f for f in bin_to_num_group[bf] if f <= bf]
                        stack.append((mytree.left_child(node),
                                      node_depth + 1,
                                      [f for f in useable_features if f not in unuseable_left]))
                        stack.append((mytree.right_child(node),
                                      node_depth + 1,
                                      [f for f in useable_features if f not in unuseable_right]))
                    else:
                        stack.append((mytree.left_child(node),
                                      node_depth + 1,
                                      useable_features))
                        stack.append((mytree.right_child(node),
                                      node_depth + 1,
                                      useable_features))


                elif node_depth == max_depth:
                    # Otherwise we have reached a true leaf node, simple reuse the predicted class from the root
                    leaf_prediction[node] = root_pred
                else:
                    print('????')

    theta = [0] * n_samples
    theta_per_leaf = {(n,i): 0 for i in I for n in mytree.L}

    samples_in_node = {n: [] for n in mytree.B + mytree.L}

    # Follow each sample down the tree, tracking the decision path and checking if it is classified correctly
    DecisionPaths = []
    for i in I:
        node = 1
        path = []
        while node in mytree.B:
            path.append(node)
            samples_in_node[node].append(i)
            f = branch_feature[node]
            if X[i,f] == 1:
                node = mytree.right_child(node)
            else:
                node = mytree.left_child(node)

        assert node in mytree.L

        path.append(node)
        samples_in_node[node].append(i)

        if y[i] == leaf_prediction[node]:
            theta[i] = 1
            theta_per_leaf[node,i] = 1

        DecisionPaths.append(path)

    b = {(n,f): 1 if branch_feature[n] == f else 0
         for n in mytree.B for f in F}
    w = {(k,n): 1 if leaf_prediction[n] == k else 0
         for n in mytree.L for k in K}

    soln_dict = {}

    if 'CART polish solutions' in opts:
        b_subtrees,w_subtrees,theta_polished = optimise_subtrees(X,y,samples_in_node,mytree,opts,branch_feature,
                                                                 cat_feature_maps=cat_feature_maps, num_feature_maps=num_feature_maps)

        if b_subtrees is not None and sum(theta_polished) >= sum(theta):
            b |= b_subtrees
            w |= w_subtrees
            soln_dict['theta old'] = theta
            soln_dict['theta'] = theta_polished
            theta = theta_polished

            # Need to update the decision paths if we later want to get the flow variables
            if 'CART flow vars' in opts:
                branch_feature = {}
                for n in mytree.B:
                    for f in F:
                        if b[n,f] == 1:
                            branch_feature[n] = f
                            break
                # Follow each sample down the tree, tracking the decision path and checking if it is classified correctly
                DecisionPaths = []
                for i in I:
                    node = 1
                    path = []
                    while node in mytree.B:
                        path.append(node)
                        f = branch_feature[node]
                        if X[i, f] == 1:
                            node = mytree.right_child(node)
                        else:
                            node = mytree.left_child(node)

                    path.append(node)
                    DecisionPaths.append(path)


    soln_dict['b'] = b
    soln_dict['w'] = w
    soln_dict['theta'] = theta

    if 'CART flow vars' in opts:

        # Set all edges to zero
        z = {(n1,n2,i): 0
             for n1 in mytree.B for n2 in mytree.children(n1) for i in I}

        for i in I:
            z[(mytree.source, 1, i)] = 0
            for n in mytree.L:
                z[(n, mytree.sink, i)] = 0

        # Fill in edges which have a flow
        for i in I:
            if theta[i] > 0.5:
                z[(mytree.source, 1, i)] = 1
                path = DecisionPaths[i]
                for j in range(len(path)-1):
                    z[path[j],path[j+1],i] = 1
                z[(path[-1], mytree.sink, i)] = 1

        soln_dict['z'] = z

    return soln_dict

def optimise_depth2_subtree(X,y,
                                 tree=None,
                                 weights=None,
                                 invalid_features=None,
                                 bin_to_cat_group=None,
                                 bin_to_num_group=None):

    # TODO: Allow for weights to be used
    if weights is not None:
        assert sum(weights) == len(weights)
    # assert bin_to_cat_group is None
    # assert bin_to_num_group is None

    # X - feature values
    # y - sample classes
    # K - Possible target classed. NOTE: may not coincide with np.unique(y) since
    # y could theoretically be a subset of the dataset with

    n_samples, n_features = X.shape
    I, F_all = range(n_samples), range(n_features)

    # Get unique classes in data and associate them to an index
    # Need k_to_class_idx for when when K != [0,1,...,|K|]
    K = np.unique(y).tolist()
    k_to_class_idx = {k: i for i, k in enumerate(K)}

    # If no tree is given create a generic depth 2 tree
    if tree is None:
        tree = Tree(2)
    elif tree.depth != 2:
        return None, None, None

    if weights is None:
        weights = [1] * n_samples

    if invalid_features is not None:
    # Given a set of invalid features, we want the subroutine to operate only on the subset of valid features
    # and then map back to the full set of features

        all_map_to_F = {}
        F_map_to_all = []   # Keep track of how our reduced feature set maps back to all features
        F_mask = []         # Used to mask the feature dimension when indexing X

        for i, f in enumerate(F_all):
            if f not in invalid_features:
                all_map_to_F[f] = len(F_map_to_all)
                F_map_to_all.append(i)
                F_mask.append(True)
            else:
                F_mask.append(False)

        F = range(len(F_map_to_all))
    else:
        F_map_to_all = F_all
        F_mask = [True] * len(F_all)
        F = F_all

    # bin_to_cat_group_mapped = {all_map_to_F[]}


    # Subroutine will fail with no data supplied
    if n_samples == 0:
        w = [0, 0, 0, 0]
        theta_idx = []

        # Still assure that output conforms to NoFeatureReuse inequalities if required
        if bin_to_cat_group is not None or bin_to_num_group is not None:
            soln_found = False
            for f in F:
                parent_feature = F_map_to_all[f]

                # Find invalid left and right children for the given candidate parent feature
                if bin_to_num_group is not None and parent_feature in bin_to_num_group:
                    left_invalid_grouping = set(f for f in bin_to_num_group[parent_feature] if f >= parent_feature)
                    right_invalid_grouping = set(f for f in bin_to_num_group[parent_feature] if f <= parent_feature)
                elif bin_to_cat_group is not None and parent_feature in bin_to_num_group:
                    left_invalid_grouping = [parent_feature]
                    right_invalid_grouping = bin_to_num_group[parent_feature]
                else:
                    left_invalid_grouping = [parent_feature]
                    right_invalid_grouping = [parent_feature]

                left_child_feature, right_child_feature = None, None

                for f_left in F:
                    f_left_candidate = F_map_to_all[f_left]
                    if f_left_candidate not in left_invalid_grouping:
                        left_child_feature = f_left_candidate

                for f_right in F:
                    f_right_candidate = F_map_to_all[f_right]
                    if f_right_candidate not in right_invalid_grouping:
                        right_child_feature = f_right_candidate

                if left_child_feature is not None and right_child_feature is not None:
                    b = (parent_feature, left_child_feature, right_child_feature)
                    soln_found = True
                    break

            if not soln_found:
                # In this case there are no feature combinations which are valid according to the
                # provided invalid_feature set and feature mappings
                return None, None, None

        else:
            b = (F_map_to_all[0], F_map_to_all[0], F_map_to_all[0])

        return b, w, theta_idx

    ############### BEGIN SUBROUTINE PROPER ###############

    # # Set up frequency counters
    # FQ_ref = {'0': [[0 for d1 in F] for d0 in range(len(K))],
    #       '1': [[0 for d1 in F] for d0 in range(len(K))],
    #       '00': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))],
    #       '01': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))],
    #       '10': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))],
    #       '11': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))]}

    # TODO: Cut down on memory usage by using smaller ints?
    FQ1 = np.zeros((len(K), len(F)), np.int32)
    FQ11 = np.zeros((len(K), len(F), len(F)), np.int32)

    # Track number of samples of each class in the subtree
    D_ref = [0] * len(K)

    _, D = np.unique(y, return_counts=True)

    # for idx in I:
    #     k = k_to_class_idx[y[idx]]   # Offset to index into lists
    #     fv = X[idx,F_mask]     # feature vector
    #     D_ref[k] += 1
    #     for i in F:
    #         if fv[i] > 0.5:
    #             FQ_ref['1'][k][i] += weights[idx]
    #             for j in F:
    #                 if fv[j] > 0.5:
    #                     FQ_ref['11'][k][i][j] += weights[idx]

    for k in K:
        # Take the subarray of X in which all samples have class y^i = k and features are filtered by F_mask
        X_masked = X[np.ix_(y==k,F_mask)]

        k_idx = k_to_class_idx[k]

        # Each column of X_masked is associated with feature f, and indicates which samples have x_f^i == 1
        # Taking the dot product of columns associated with features f_a and f_b will be equal to the number of
        # samples for which (x_fa^i == 1 AND x_fb^i == 1)
        FQ11[k_idx,:,:] = X_masked.T @ X_masked

        # The diagonal corresponds to duplicated features, i.e. x_fa^i == 1
        FQ1[k_idx,:] = np.diag(FQ11[k_idx,:,:])


    # assert (np.sum(np.asarray(FQ_ref['1']) != FQ1) == 0)
    # assert (np.sum(np.asarray(FQ_ref['11']) != FQ11) == 0)
    #
    # # Fill out symmetry of matrix
    # for k in range(len(K)):
    #     for i in F:
    #         for j in range(i+1,len(F)):
    #             FQ_ref['11'][k][j][i] = FQ_ref['11'][k][i][j]
    #
    # for k in range(len(K)):
    #     for i in F:
    #         FQ_ref['0'][k][i] = D[k] - FQ_ref['1'][k][i]
    #         for j in F:
    #             FQ_ref['10'][k][i][j] = FQ_ref['1'][k][i] - FQ_ref['11'][k][i][j]
    #             FQ_ref['01'][k][i][j] = FQ_ref['1'][k][j] - FQ_ref['11'][k][i][j]
    #             FQ_ref['00'][k][i][j] = FQ_ref['0'][k][i] - FQ_ref['01'][k][i][j]

    FQ0 = np.expand_dims(D,axis=1) - FQ1
    FQ10 = np.expand_dims(FQ1,axis=2) - FQ11
    FQ01 = np.expand_dims(FQ1,axis=1) - FQ11
    FQ00 = np.expand_dims(FQ0,axis=2) - FQ01

    # assert (np.sum(np.asarray(FQ_ref['0']) != FQ0) == 0)
    # assert (np.sum(np.asarray(FQ_ref['10']) != FQ10) == 0)
    # assert (np.sum(np.asarray(FQ_ref['01']) != FQ01) == 0)
    # assert (np.sum(np.asarray(FQ_ref['00']) != FQ00) == 0)
    #
    # leaves = ['00', '01', '10', '11']

    # Use frequency counters to determine the optimal subtree structure

    # # Number correctly classified in each leaf for each combination of features
    # num_classified_ref = {c: [[0 for d1 in F] for d0 in F]
    #                   for c in leaves}
    # best_class = {c: [[None for d1 in F] for d0 in F]
    #                   for c in leaves}
    #
    # for i in F:
    #     for j in F:
    #         for leaf in leaves:
    #             best_class_idx = None
    #             best_class_obj = -1
    #             for k in range(len(K)):
    #                 num_in_leaf = FQ_ref[leaf][k][i][j]
    #                 if num_in_leaf > best_class_obj:
    #                     best_class_obj = num_in_leaf
    #                     best_class_idx = k
    #             num_classified_ref[leaf][i][j] = best_class_obj
    #             best_class[leaf][i][j] = K[best_class_idx]

    # For each combination of features, determine in each leaf
    # the number of samples with the majority class
    n_classified00 = np.max(FQ00,axis=0)
    n_classified01 = np.max(FQ01, axis=0)
    n_classified10 = np.max(FQ10, axis=0)
    n_classified11 = np.max(FQ11, axis=0)

    # assert (np.sum(np.asarray(num_classified_ref['00']) != n_classified00) == 0)
    # assert (np.sum(np.asarray(num_classified_ref['01']) != n_classified01) == 0)
    # assert (np.sum(np.asarray(num_classified_ref['10']) != n_classified10) == 0)
    # assert (np.sum(np.asarray(num_classified_ref['11']) != n_classified11) == 0)
    #
    # left_subtree_scores_ref = [[num_classified_ref['00'][i][j] + num_classified_ref['01'][i][j] for j in F] for i in F]
    # right_subtree_scores_ref = [[num_classified_ref['10'][i][j] + num_classified_ref['11'][i][j] for j in F] for i in F]

    # Get the number of samples correctly classified in the left and right subtrees for each feature combination
    # dimension (|F|,|F|) where element (a,b) is the objective for parent feature f_a with left/right child feature f_b
    left_subtree_scores = n_classified00 + n_classified01
    right_subtree_scores = n_classified10 + n_classified11

    # assert (np.sum(np.asarray(left_subtree_scores_ref) != left_subtree_scores) == 0)
    # assert (np.sum(np.asarray(right_subtree_scores_ref) != right_subtree_scores) == 0)
    #
    # best_obj_ref = -1
    # best_parent_feature_ref = None
    #
    # for i in F:
    #     left_obj, right_obj = -1, -1
    #     left_features = []
    #     right_features = []
    #     for j in F:
    #         left_score = left_subtree_scores_ref[i][j]
    #         right_score = right_subtree_scores_ref[i][j]
    #
    #         if left_score > left_obj:
    #             left_features = [j]
    #             left_obj = left_subtree_scores_ref[i][j]
    #         elif left_score == left_obj:
    #             left_features.append(j)
    #
    #         if right_score > right_obj:
    #             right_features = [j]
    #             right_obj = right_subtree_scores_ref[i][j]
    #         elif right_score == right_obj:
    #             right_features.append(j)
    #
    #     if left_obj + right_obj > best_obj_ref:
    #         best_obj_ref = left_obj + right_obj
    #         best_parent_feature_ref = i
    #         best_left_child_features_ref = left_features
    #         best_right_child_features_ref = right_features

    # Find the combination of parent/child features that maximise the
    # number of correctly classified points

    subtree_left_maxes = np.max(left_subtree_scores, axis=1)
    subtree_right_maxes = np.max(right_subtree_scores, axis=1)

    total_scores = subtree_left_maxes + subtree_right_maxes

    # TODO: Investigate multiple potential best parent features simultaneously?
    best_parent_feature = np.argmax(total_scores)

    if bin_to_cat_group is not None or bin_to_num_group is not None:
        # If we are provided with categorical or numerical feature groupings we
        # must make sure that the solution respects them

        # Find the best objective in each subtree for the given parent feature
        best_left_subtree_objective = np.max(left_subtree_scores[best_parent_feature,:])
        best_right_subtree_objective = np.max(right_subtree_scores[best_parent_feature, :])

        # In each subtree find the set of branch features which give the maximum objective
        candidate_left_features = np.nonzero(left_subtree_scores[best_parent_feature,:] == best_left_subtree_objective)[0]
        candidate_right_features = np.nonzero(right_subtree_scores[best_parent_feature, :] == best_right_subtree_objective)[0]

        # We want the features which are made invalid in the left and right subtrees by the parent feature
        parent_feature_all = F_map_to_all[best_parent_feature]

        if bin_to_num_group is not None and parent_feature_all in bin_to_num_group:
            all_num_group = bin_to_num_group[parent_feature_all]
            F_num_group = [all_map_to_F[f] for f in all_num_group if f in all_map_to_F]

            left_invalid_grouping = [f for f in F_num_group if f >= best_parent_feature]
            right_invalid_grouping = [f for f in F_num_group if f <= best_parent_feature]

        elif bin_to_cat_group is not None and parent_feature_all in bin_to_cat_group:
            all_cat_group = bin_to_cat_group[parent_feature_all]    # Binary features associated parent feature categorical variable
            F_cat_group = [all_map_to_F[f] for f in all_cat_group if f in all_map_to_F]     # Associated binary features in the REDUCED feature space

            left_invalid_grouping = [best_parent_feature]
            right_invalid_grouping = F_cat_group

        else:
            left_invalid_grouping = [best_parent_feature]
            right_invalid_grouping = [best_parent_feature]

        # Filter the feature choices which give maximum objectives in each subtree based on the supplied binary feature mappings
        candidate_left_features = candidate_left_features[~np.isin(candidate_left_features, left_invalid_grouping)]
        candidate_right_features = candidate_right_features[~np.isin(candidate_right_features, right_invalid_grouping)]

        if len(candidate_left_features) > 0 and len(candidate_right_features) > 0:
            best_left_child_feature = candidate_left_features[0]
            best_right_child_feature = candidate_right_features[0]
        else:
            print('Could not find optimal depth 2 subtree which satisfies \"No Feature Reuse\" option')
            best_left_child_feature = np.argmax(left_subtree_scores[best_parent_feature, :])
            best_right_child_feature = np.argmax(right_subtree_scores[best_parent_feature, :])
    else:
        best_left_child_feature = np.argmax(left_subtree_scores[best_parent_feature,:])
        best_right_child_feature = np.argmax(right_subtree_scores[best_parent_feature, :])




    # if bin_to_cat_group is not None or bin_to_num_group is not None:
    #     # If we are provided with categorical or numerical feature groupings we
    #     # must make sure that the solution respects them
    #     best_left_child_feature_ref = None
    #     best_right_child_feature = None
    #     for j in best_left_child_features_ref:
    #         # If invalid features were supplied, maps back to the full set of feature F_all
    #         left_feature_all = F_map_to_all[j]
    #         parent_feature_all = F_map_to_all[best_parent_feature_ref]
    #
    #         if bin_to_num_group is not None and parent_feature_all in bin_to_num_group:
    #             invalid_grouping = set(f for f in bin_to_num_group[parent_feature_all] if f >= parent_feature_all)
    #         else:
    #             invalid_grouping = [parent_feature_all]
    #
    #
    #         if left_feature_all not in invalid_grouping:
    #             best_left_child_feature_ref = j
    #             break
    #
    #     for j in best_right_child_features_ref:
    #         # If invalid features were supplied, maps back to the full set of feature F_all
    #         right_feature_all = F_map_to_all[j]
    #         parent_feature_all = F_map_to_all[best_parent_feature_ref]
    #
    #         if bin_to_cat_group is not None and parent_feature_all in bin_to_cat_group:
    #             invalid_grouping = bin_to_cat_group[parent_feature_all]
    #         if bin_to_num_group is not None and parent_feature_all in bin_to_num_group:
    #             invalid_grouping = set(f for f in bin_to_num_group[parent_feature_all] if f <= parent_feature_all)
    #
    #         if right_feature_all not in invalid_grouping:
    #             best_right_child_feature = j
    #             break
    #
    #     if best_left_child_feature_ref is None:
    #         best_left_child_feature_ref = best_left_child_features_ref[0]
    #         print('Could not find optimal depth 2 subtree which satisfies \"No Feature Reuse\" option')
    #
    #     if best_right_child_feature is None:
    #         best_right_child_feature = best_right_child_features_ref[0]
    #         print('Could not find optimal depth 2 subtree which satisfies \"No Feature Reuse\" option')
    #
    # else:
    #     best_left_child_feature_ref = best_left_child_features_ref[0]
    #     best_right_child_feature_ref = best_right_child_features_ref[0]

    # assert (best_parent_feature == best_parent_feature_ref)
    # assert (best_left_child_feature == best_left_child_feature_ref)
    # assert (best_right_child_feature == best_right_child_feature_ref)
    #
    #
    # leaf_predictions = {'00': best_class['00'][best_parent_feature_ref][best_left_child_feature],
    #                     '01': best_class['01'][best_parent_feature_ref][best_left_child_feature],
    #                     '10': best_class['10'][best_parent_feature_ref][best_right_child_feature],
    #                     '11': best_class['11'][best_parent_feature_ref][best_right_child_feature]}


    # # Map from reduced feature set back to all features
    # best_parent_feature_ref = F_map_to_all[best_parent_feature_ref]
    # best_left_child_feature_ref = F_map_to_all[best_left_child_feature_ref]
    # best_right_child_feature_ref = F_map_to_all[best_right_child_feature_ref]
    #
    # b_ref = (best_parent_feature_ref, best_left_child_feature_ref, best_right_child_feature_ref)
    # w_ref = [leaf_predictions[leaf] for leaf in leaves]
    # theta = []

    w_class_idx = [np.argmax(FQ00[:, best_parent_feature, best_left_child_feature]),
                   np.argmax(FQ01[:, best_parent_feature, best_left_child_feature]),
                   np.argmax(FQ10[:, best_parent_feature, best_right_child_feature]),
                   np.argmax(FQ11[:, best_parent_feature, best_right_child_feature])]

    # Convert back to original classes, instead of the idx in the sorted list of unique classes
    w = [K[k_idx] for k_idx in w_class_idx]

    best_parent_feature = F_map_to_all[best_parent_feature]
    best_left_child_feature = F_map_to_all[best_left_child_feature]
    best_right_child_feature = F_map_to_all[best_right_child_feature]

    b = (best_parent_feature, best_left_child_feature, best_right_child_feature)

    # assert b == b_ref
    # assert w == w_ref

    # theta_ref = []
    #
    # for idx in I:
    #     sample = X[idx,:]
    #     if sample[best_parent_feature] > 0.5:
    #         # Sample goes right
    #         if sample[best_right_child_feature] > 0.5:
    #             if y[idx] == w[3]:
    #                 theta_ref.append(idx)
    #         else:
    #             if y[idx] == w[2]:
    #                 theta_ref.append(idx)
    #     else:
    #         # Sample goes left
    #         if sample[best_left_child_feature] > 0.5:
    #             if y[idx] == w[1]:
    #                 theta_ref.append(idx)
    #         else:
    #             if y[idx] == w[0]:
    #                 theta_ref.append(idx)

    theta = np.zeros(n_samples,dtype=bool)

    left_subtree_mask = (X[:,best_parent_feature] == 0)
    right_subtree_mask = (X[:, best_parent_feature] == 1)

    leaf00_mask = left_subtree_mask * (X[:,best_left_child_feature] == 0) * (y == w[0])
    leaf01_mask = left_subtree_mask * (X[:, best_left_child_feature] == 1) * (y == w[1])
    leaf10_mask = right_subtree_mask * (X[:, best_right_child_feature] == 0) * (y == w[2])
    leaf11_mask = right_subtree_mask * (X[:, best_right_child_feature] == 1) * (y == w[3])

    theta[leaf00_mask + leaf01_mask + leaf10_mask + leaf11_mask] = True
    theta = np.nonzero(theta)[0]

    # assert np.sum(np.asarray(theta_ref) != theta) == 0

    # theta[leaf00_mask] = True
    # theta[leaf01_mask] = True
    # theta[leaf10_mask] = True
    # theta[leaf11_mask] = True

    return b, w, theta

def optimise_depth2_subtree_reference(X,y,
                            tree=None,
                            weights=None,
                            invalid_features=None,
                            bin_to_cat_group=None,
                            bin_to_num_group=None):
    # X - feature values
    # y - sample classes

    # Want following options:
    # Set invalid features which cannot be used. Could do this outside and filter X?
    # Give invalid feature combinations. E.g. branching on a feature may imply we can not longer
    # branch on a specific set of features in the left or right subtrees

    n_samples, n_features = X.shape
    I, F_all = range(n_samples), range(n_features)

    # Get unique classes in data and associate them to an index
    K = np.unique(y).tolist()
    k_to_class_idx = {k: i for i, k in enumerate(K)}

    # If no tree is given create a generic depth 2 tree
    if tree is None:
        tree = Tree(2)
    elif tree.depth != 2:
        return None, None, None

    if weights is None:
        weights = [1] * n_samples

    if invalid_features is not None:
    # Given a set of invalid features, we want the subroutine to operate only on the subset of valid features
    # and then map back to the full set of features

        F_map_to_all = []   # Keep track of how our reduced feature set maps back to all features
        F_mask = []         # Used to mask the feature dimension when indexing X

        for i, f in enumerate(F_all):
            if f not in invalid_features:
                F_map_to_all.append(i)
                F_mask.append(True)
            else:
                F_mask.append(False)

        F = range(len(F_map_to_all))
    else:
        F_map_to_all = F_all
        F_mask = [True] * len(F_all)
        F = F_all

    # Subroutine will fail with no data supplied
    if n_samples == 0:
        w = [0, 0, 0, 0]
        theta_idx = []

        # Still assure that output conforms to NoFeatureReuse inequalities if required
        if bin_to_cat_group is not None or bin_to_num_group is not None:
            for f in F:
                parent_feature = F_map_to_all[f]

                # Find invalid left and right children for the given candidate parent feature
                if bin_to_num_group is not None and parent_feature in bin_to_num_group:
                    left_invalid_grouping = set(f for f in bin_to_num_group[parent_feature] if f >= parent_feature)
                    left_invalid_grouping = set(f for f in bin_to_num_group[parent_feature] if f <= parent_feature)
                elif bin_to_cat_group is not None and parent_feature in bin_to_num_group:
                    left_invalid_grouping = [parent_feature]
                    right_invalid_grouping = bin_to_num_group[parent_feature]
                else:
                    left_invalid_grouping = [parent_feature]
                    right_invalid_grouping = [parent_feature]

                left_child_feature, right_child_feature = None, None

                for f_left in F:
                    f_left_candidate = F_map_to_all[f_left]
                    if f_left_candidate not in left_invalid_grouping:
                        left_child_feature = f_left_candidate

                for f_right in F:
                    f_right_candidate = F_map_to_all[f_right]
                    if f_right_candidate not in right_invalid_grouping:
                        right_child_feature = f_right_candidate

                if left_child_feature is not None and right_child_feature is not None:
                    b = (parent_feature, left_child_feature, right_child_feature)

        else:
            b = (F_map_to_all[0], F_map_to_all[0], F_map_to_all[0])

        return b, w, theta_idx

    ############### BEGIN SUBROUTINE PROPER ###############

    # Set up frequency counters
    FQ = {'0': [[0 for d1 in F] for d0 in range(len(K))],
          '1': [[0 for d1 in F] for d0 in range(len(K))],
          '00': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))],
          '01': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))],
          '10': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))],
          '11': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))]}


    # Track number of samples of each class in the subtree
    D = [0] * len(K)

    for idx in I:
        k = k_to_class_idx[y[idx]]   # Offset to index into lists
        fv = X[idx,F_mask]     # feature vector
        D[k] += 1
        for i in F:
            if fv[i] > 0.5:
                FQ['1'][k][i] += weights[idx]
                for j in F:
                    if fv[j] > 0.5:
                        FQ['11'][k][i][j] += weights[idx]

    # Fill out symmetry of matrix
    for k in range(len(K)):
        for i in F:
            for j in range(i+1,len(F)):
                FQ['11'][k][j][i] = FQ['11'][k][i][j]

    for k in range(len(K)):
        for i in F:
            FQ['0'][k][i] = D[k] - FQ['1'][k][i]
            for j in F:
                FQ['10'][k][i][j] = FQ['1'][k][i] - FQ['11'][k][i][j]
                FQ['01'][k][i][j] = FQ['1'][k][j] - FQ['11'][k][i][j]
                FQ['00'][k][i][j] = FQ['0'][k][i] - FQ['01'][k][i][j]
                # FQ2['00'][k][i][j] = D[k] - FQ2['1'][k][i] - FQ2['1'][k][j] + FQ2['11'][k][i][j]


    leaves = ['00', '01', '10', '11']

    # Use frequency counters to determine the optimal subtree structure

    # Number correctly classified in each leaf for each combination of features
    num_classified = {c: [[0 for d1 in F] for d0 in F]
                      for c in leaves}
    best_class = {c: [[None for d1 in F] for d0 in F]
                      for c in leaves}

    for i in F:
        for j in F:
            for leaf in leaves:
                best_class_idx = None
                best_class_obj = -1
                for k in range(len(K)):
                    num_in_leaf = FQ[leaf][k][i][j]
                    if num_in_leaf > best_class_obj:
                        best_class_obj = num_in_leaf
                        best_class_idx = k
                num_classified[leaf][i][j] = best_class_obj
                best_class[leaf][i][j] = K[best_class_idx]

    left_subtree_scores = [[num_classified['00'][i][j] + num_classified['01'][i][j] for j in F] for i in F]
    right_subtree_scores = [[num_classified['10'][i][j] + num_classified['11'][i][j] for j in F] for i in F]

    best_obj = -1
    best_parent_feature = None

    for i in F:
        left_obj, right_obj = -1, -1
        left_features = []
        right_features = []
        for j in F:
            left_score = left_subtree_scores[i][j]
            right_score = right_subtree_scores[i][j]

            if left_score > left_obj:
                left_features = [j]
                left_obj = left_subtree_scores[i][j]
            elif left_score == left_obj:
                left_features.append(j)

            if right_score > right_obj:
                right_features = [j]
                right_obj = right_subtree_scores[i][j]
            elif right_score == right_obj:
                right_features.append(j)

        if left_obj + right_obj > best_obj:
            best_obj = left_obj + right_obj
            best_parent_feature = i
            best_left_child_features = left_features
            best_right_child_features = right_features

    if bin_to_cat_group is not None or bin_to_num_group is not None:
        # If we are provided with categorical or numerical feature groupings we
        # must make sure that the solution respects them
        best_left_child_feature = None
        best_right_child_feature = None
        for j in best_left_child_features:
            # If invalid features were supplied, maps back to the full set of feature F_all
            left_feature_all = F_map_to_all[j]
            parent_feature_all = F_map_to_all[best_parent_feature]

            if bin_to_num_group is not None and parent_feature_all in bin_to_num_group:
                invalid_grouping = set(f for f in bin_to_num_group[parent_feature_all] if f >= parent_feature_all)
            else:
                invalid_grouping = [parent_feature_all]


            if j not in invalid_grouping:
                best_left_child_feature = j
                break

        for j in best_right_child_features:
            # If invalid features were supplied, maps back to the full set of feature F_all
            right_feature_all = F_map_to_all[j]
            parent_feature_all = F_map_to_all[best_parent_feature]

            if bin_to_cat_group is not None and parent_feature_all in bin_to_cat_group:
                invalid_grouping = bin_to_cat_group[parent_feature_all]
            if bin_to_num_group is not None and parent_feature_all in bin_to_num_group:
                invalid_grouping = set(f for f in bin_to_num_group[parent_feature_all] if f <= parent_feature_all)

            if right_feature_all not in invalid_grouping:
                best_right_child_feature = j
                break

        if best_left_child_feature is None:
            best_left_child_feature = best_left_child_features[0]
            print('Could not find optimal depth 2 subtree which satisfies \"No Feature Reuse\" option')

        if best_right_child_feature is None:
            best_right_child_feature = best_right_child_features[0]
            print('Could not find optimal depth 2 subtree which satisfies \"No Feature Reuse\" option')

    else:
        best_left_child_feature = best_left_child_features[0]
        best_right_child_feature = best_right_child_features[0]


    # Update the dictionaries which hold the tree structure
    left_child, right_child = tree.children(tree.root)

    leaf_predictions = {'00': best_class['00'][best_parent_feature][best_left_child_feature],
                        '01': best_class['01'][best_parent_feature][best_left_child_feature],
                        '10': best_class['10'][best_parent_feature][best_right_child_feature],
                        '11': best_class['11'][best_parent_feature][best_right_child_feature]}

    # leaf_nodes_map = {'00': tree.left_child(left_child),
    #                   '01': tree.right_child(left_child),
    #                   '10': tree.left_child(right_child),
    #                   '11': tree.right_child(right_child)}


    # Map from reduced feature set back to all features
    best_parent_feature = F_map_to_all[best_parent_feature]
    best_left_child_feature = F_map_to_all[best_left_child_feature]
    best_right_child_feature = F_map_to_all[best_right_child_feature]

    b = (best_parent_feature,best_left_child_feature,best_right_child_feature)
    w = [leaf_predictions[leaf] for leaf in leaves]
    theta = []


    for idx in I:
        sample = X[idx,:]
        if sample[best_parent_feature] > 0.5:
            # Sample goes right
            if sample[best_right_child_feature] > 0.5:
                if y[idx] == leaf_predictions['11']:
                    theta.append(idx)
            else:
                if y[idx] == leaf_predictions['10']:
                    theta.append(idx)
        else:
            # Sample goes left
            if sample[best_left_child_feature] > 0.5:
                if y[idx] == leaf_predictions['01']:
                    theta.append(idx)
            else:
                if y[idx] == leaf_predictions['00']:
                    theta.append(idx)

    return b, w, theta

def optimise_subtrees(X,y,samples_in_node,tree,opts,branch_features,
                      cache=None,weights=None,cat_feature_maps=None,num_feature_maps=None):

    n_samples, n_features = X.shape
    F_all = range(n_features)
    K = np.unique(y).tolist()

    # Double check that the tree is large enough
    if tree.depth < 2:
        return None, None, None

    # By default all samples have equal weight
    if weights is None:
        weights = [1] * n_samples

    # Tree layers stored with 0 based indexing with leaves at position -1
    subtree_roots = tree.layers[-3]

    b = {(n,f): 0
         for n in subtree_roots + tree.layers[-2] for f in F_all}
    w = {(k,n): 0
         for k in K for n in tree.L}
    theta = [0] * n_samples

    for root in subtree_roots:
        run_subroutine = True
        if cache is not None:
            assert isinstance(cache,dict)

            root_ancestors = tree.ancestors(root, branch_dirs=True)

            path_key = frozenset(str(branch_features[node]) + str(dir) for node, dir in root_ancestors.items())

            # If we have already seen this subtree, reuse the results
            if path_key in cache:
                b_subtree, w_subtree, theta_subtree = cache[path_key]
                run_subroutine = False

        if run_subroutine:
            invalid_features = set()

            if 'No Feature Reuse' in opts:
                if cat_feature_maps is None and num_feature_maps is None:
                    print('No Feature Reuse option enabled for primal heuristic feature but feature groupings not supplied')
                else:
                    bin_to_cat_group = {}
                    bin_to_num_group = {}
                    if cat_feature_maps is not None:
                        for Cf in cat_feature_maps:
                            CF_set = set(Cf)
                            for f in Cf:
                                bin_to_cat_group[f] = CF_set

                    if num_feature_maps is not None:
                        for Nf in num_feature_maps:
                            NF_set = set(Nf)
                            for f in Nf:
                                bin_to_num_group[f] = NF_set

                    for n_a, dir in tree.ancestors(root, branch_dirs=True).items():
                        ancestor_branch_feature = branch_features[n_a]
                        # Ancestor branches right
                        if dir == 1:
                            if ancestor_branch_feature in bin_to_cat_group:
                                invalid_grouping = bin_to_cat_group[ancestor_branch_feature]
                            elif ancestor_branch_feature in bin_to_num_group:
                                invalid_grouping = [f for f in bin_to_num_group[ancestor_branch_feature] if f <= ancestor_branch_feature]

                        elif dir == 0:
                            if ancestor_branch_feature in bin_to_num_group:
                                invalid_grouping = [f for f in bin_to_num_group[ancestor_branch_feature] if f >= ancestor_branch_feature]
                            else:
                                invalid_grouping = [ancestor_branch_feature]

                        for f in invalid_grouping:
                            invalid_features.add(f)

            else:
                bin_to_cat_group, bin_to_num_group = None, None

            # subtree = tree.subtree(root, root_at_n=True)
            I_root = samples_in_node[root]

            b_subtree, w_subtree , theta_subtree = optimise_depth2_subtree(X[I_root,:], y[I_root],
                                                                           weights=[weights[i] for i in I_root],
                                                                           invalid_features=invalid_features,
                                                                           bin_to_cat_group=bin_to_cat_group,
                                                                           bin_to_num_group=bin_to_num_group)

            # Cache results of subroutine
            if cache is not None:
                cache[path_key] = (b_subtree,
                                   w_subtree,
                                   theta_subtree)

        parent_feature, left_feature, right_feature = b_subtree

        # Update the dictionaries which hold the tree structure
        left_child, right_child = tree.children(root)

        b[root, parent_feature] = 1
        b[left_child, left_feature] = 1
        b[right_child, right_feature] = 1

        _, subtree_leaf_nodes = tree.descendants(root, split_nodes=True)
        for i, n in enumerate(subtree_leaf_nodes):
            w[w_subtree[i],n] = 1

        # Cached_theta_idx holds indexes of samples_in_node[root], not I.
        I_root = samples_in_node[root]
        for idx in theta_subtree:
            theta[I_root[idx]] = 1

    return b,w,theta

class Tree():
    def __init__(self,depth,root=1):
        self.depth = depth
        self.root = root

        self.B = list(range(1, 2 ** depth))
        self.L = list(range(2 ** depth, 2 ** (depth + 1)))
        self.T = self.B + self.L
        self.source = 0
        self.sink = 2**(depth + 1)
        self.layers = [list(range(2 ** d, 2 ** (d + 1))) for d in range(depth+1)]

    def subtree(self,n,root_at_n=False):
        # Returns a tree object rooted at node
        # By default sets the root node at n=1. Maybe if it's useful I can add an option to retain the node numbering

        # TODO: Do this properly by rewriting __init__. This should work fine for now (except for self.layers, source and sink)

        if root_at_n:
            new_subtree = Tree(self.height(n), root=n)
            new_subtree.B, new_subtree.L = self.descendants(n, split_nodes=True)
        else:
            return Tree(self.height(n))

    def left_child(self,n):
        if n in self.L:
            return None
        else:
            return 2*n

    def right_child(self,n):
        if n in self.L:
            return None
        else:
            return 2*n + 1

    def children(self,n):
        left_child = self.left_child(n)

        # Return empty list if leaf node (no children)
        if left_child is None:
            return []

        right_child = self.right_child(n)

        return [left_child, right_child]

    def parent(self,n):
        # Returns the parent node of a given branch node
        # Special case at root node which has the source node as a parent
        if n==1:
            return 0

        # If even number the node is a left child of the parent.
        # If odd number node is a right child
        if n%2 == 0:
            return n//2
        else:
            return (n-1)//2

    # TODO: Rewrite with recursion?
    def ancestors(self,n,branch_dirs=False):

        # Stores a dict with ancestor nodes in the keys and the branch direction (0 = left, 1 = right) of the child as a value
        # E.g. if node n is a left child then node_ancestors[parent_node(n)] = 0
        node_ancestors = {}
        left_path = []
        right_path = []

        # Work up ancestors until root node
        while n > 1:
            # Even number -> left child of parent
            # Odd number -> right child of parent
            branch = 0 if (n % 2 == 0) else 1
            n = n // 2
            node_ancestors[n] = branch

        if branch_dirs:
            return node_ancestors
        else:
            return list(node_ancestors.keys())

    #TODO: Rewrite to make less awkward
    def descendants(self,n,split_nodes=False):
        # Returns the descendants of node n, inclusive of node n
        # By defaults return branch and left nodes in one leaf
        # If split_nodes=True, returns tuple of (branch_nodes, leaf_nodes)


        node_descendants = []
        stack = [n]

        leaf_start_idx = -1

        while len(stack) > 0:
            current_node = stack.pop(0)

            if split_nodes and leaf_start_idx == -1 and current_node >= self.L[0]:
                leaf_start_idx = len(node_descendants)

            node_descendants.append(current_node)

            left_child = self.left_child(current_node)

            # left_child = None implies that we have reach leaf nodes which have no descendants
            if left_child is None:
                continue

            right_child = self.right_child(current_node)

            stack.append(left_child)
            stack.append(right_child)
        if split_nodes:
            return node_descendants[:leaf_start_idx], node_descendants[leaf_start_idx:]
        else:
            return node_descendants

    def descendant_leaves(self,root):
        # Return the leafs which are descendant from the given node
        # If root is a leaf node then it returns just the leaf node

        subtree_depth = self.height(root)
        return list(range(root * 2**subtree_depth,root * 2**subtree_depth + 2**subtree_depth))

    def height(self,node):

        node_depth = 0
        n = node

        while n != 1:
            node_depth += 1
            n = n // 2

        node_height = self.depth - node_depth
        return node_height

def find_split_sets(data,max_removed=None,force_recalc=False):
    # Given datasets (X,y) return the support (features with the same values) for each pair (i,j) such that y_i != y_j

    instance_name = data['name']
    encoding_scheme = data['encoding']

    file_dir = os.path.join('Datasets','Auxfiles')
    if encoding_scheme is not None:
        file = os.path.join(file_dir, f'{instance_name}_{encoding_scheme}_splitsets.pickle')
    else:
        file = os.path.join(file_dir, instance_name + '_splitsets.pickle')

    if not force_recalc:
        try:
            with open(file, 'rb') as f:
                split_sets = pickle.load(f)
            print(f'Loaded in split sets for {data['name']} dataset')

            if max_removed is not None:
                return [(cut_idx, bound, removed_features) for cut_idx, bound, removed_features in split_sets if len(removed_features) <= max_removed]

            return split_sets
        except:
            print(f'Failed to load in split sets for {data['name']} dataset. Attempting to recalculate')

    X, y = data['X'], data['y']

    start_time = time.time()

    n_samples, n_features = X.shape
    I = range(n_samples)
    F = range(n_features)

    if max_removed is None:
        max_fr = len(F)
    else:
        max_fr = max_removed

    eqp_cuts = {}
    support_sets = {}

    for i,j in itertools.combinations(I,2):
        if y[i] != y[j]:
            # Get subset of feature where x^i != x^j, i.e. if these features were removed then x^i == x^j
            F_support = tuple(np.nonzero(X[i,:] == X[j,:])[0])
            F_star = tuple(f for f in F if f not in F_support)
            if len(F_star) <= max_fr:
                # Check if we have already seen a pair with identical support (support features and support feature values)
                support_key = (F_support,tuple(X[i,F_support]))
                if support_key in support_sets:
                    orig_cut_idx = support_sets[support_key]
                    new_cut_idx = tuple(sorted(list(set(orig_cut_idx + (i,j)))))

                    support_sets[support_key] = new_cut_idx
                    del eqp_cuts[orig_cut_idx]

                    # Determine the new bound of the cut_idx
                    classes = {}
                    for idx in new_cut_idx:
                        if y[idx] not in classes:
                            classes[y[idx]] = 1
                        else:
                            classes[y[idx]] += 1

                    bound = max(classes.values())

                    eqp_cuts[new_cut_idx] = {'Removed Features': F_star,
                                             'Bound': bound}

                    continue

                eqp_cuts[i,j] = {'Removed Features': F_star,
                                 'Bound': 1}
                support_sets[(F_support,tuple(X[i,F_support]))] = (i,j)

    split_sets = [(cut_idx, values['Bound'], values['Removed Features']) for cut_idx, values in eqp_cuts.items()]

    with open(file, 'wb') as f:
        pickle.dump(split_sets, f)

    if max_removed is not None:
        return [(cut_idx, bound, removed_features) for cut_idx, bound, removed_features in split_sets if len(removed_features) <= max_removed]

    return split_sets

def Equivalent_Points_Orig(X,y,max_removed=1):
    _, n_samples = X.shape
    F = list(range(n_samples))

    # Get equivalent points with no features removed
    all_features_bound_moved, _, eqp_idx, eqp_max = _find_equivalent_points(X, y)

    # Elements of eqp_cuts are tuples (idx_cuts, rhs_bound, features which deactivate bound)
    eqp_cuts = []
    eqp_cuts_set = set(eqp_idx)

    for idx, rhs_bound in zip(eqp_idx, eqp_max):
        eqp_cuts.append((idx, rhs_bound, None))

    cut_info = {0: (len(eqp_cuts), all_features_bound_moved)}

    if max_removed == 0:
        return eqp_cuts, cut_info

    bound_moved_total = 0
    num_cuts_added = 0

    # Now get equivalent points for a single feature removed
    for f in F:
        removed_features = [f]
        bound_moved, _, eqp_idx, eqp_max = _find_equivalent_points(np.delete(X, removed_features, axis=1), y)

        if bound_moved > all_features_bound_moved:
            for idx, rhs_bound in zip(eqp_idx, eqp_max):
                if idx not in eqp_cuts_set:
                    eqp_cuts_set.add(idx)
                    eqp_cuts.append((idx, rhs_bound, f))
                    # num_cuts_added += 1
                    # bound_moved_total += len(eqp_idx) - rhs_bound

    cut_info[1] = (num_cuts_added,bound_moved_total)

    return eqp_cuts

def Equivalent_Points_WIP(X,y,max_removed=1):

    _, n_samples = X.shape
    F = list(range(n_samples))

    # Get equivalent points with no features removed
    all_features_bound_moved, _, eqp_idx, eqp_max = _find_equivalent_points(X,y)

    # Elements of eqp_cuts are tuples (idx_cuts, rhs_bound, features which deactivate bound)
    eqp_cuts = []
    eqp_cuts_set = set(eqp_idx)

    for idx,rhs_bound in zip(eqp_idx,eqp_max):
        eqp_cuts.append((idx, rhs_bound, []))

    bound_moved = {num_removed: {}
                   for num_removed in range(max_removed+1)}

    bound_moved[0] = {tuple(): all_features_bound_moved}

    cut_info = {0: (len(eqp_cuts), all_features_bound_moved)}

    for num_removed in range(1,max_removed+1):
        num_cuts_added = 0
        bound_moved_total = 0
        removed_feature_combos = itertools.combinations(F,num_removed)
        prev_bounds_moved_dict = bound_moved[num_removed-1]

        for removed_features in removed_feature_combos:
            bound_moved, _, eqp_idx, eqp_max = _find_equivalent_points(np.delete(X,removed_features,axis=1), y)

            # Check if we could be repeating a bound already found by removing num_removed-1 features
            removed_feature_subcombos = itertools.combinations(F,num_removed-1)
            new_cut_valid = True
            for removed_features_prev in removed_feature_subcombos:
                if bound_moved < prev_bounds_moved_dict.get(removed_features_prev,0):
                    new_cut_valid = False
                    break

            if new_cut_valid:
                for idx, rhs_bound in zip(eqp_idx, eqp_max):
                    eqp_cuts.append((idx, rhs_bound, removed_features))
                    num_cuts_added += 1
                    bound_moved_total += bound_moved

        cut_info[num_removed] = (num_cuts_added,bound_moved_total)



    return eqp_cuts, cut_info

    # for num_removed in range(max_removed+1):
    #     removed_feature_combos = itertools.combinations(F,num_removed)
    #
    #     eq_sets_found = []
    #
    #     best_bound_moved = -1
    #     num_eq_points = -1
    #     for removed_features in removed_feature_combos:
    #         bound_moved, equivalent_points_found, eqp_idx = _find_equivalent_points(np.delete(X,removed_features,axis=1),y)
    #         eq_sets_found.append((bound_moved,equivalent_points_found,removed_features,eqp_idx))
    #
    #     eq_sets_found.sort(reverse=True)

        # print(f'Removed {num_removed} features - ', eq_sets_found[min(0,len(eq_sets_found)-1)])
        # print(f'Removed {num_removed} features - ', eq_sets_found[min(1, len(eq_sets_found) - 1)])
        # print(f'Removed {num_removed} features - ', eq_sets_found[min(2, len(eq_sets_found) - 1)])

        # print(f'Removing {num_removed} features allows for a {best_bound_moved} bound shift with {num_eq_points} equivalent point sets')

def Equivalent_Points_Brute(X,y,max_removed=1,aggregate=False):
    _, n_samples = X.shape
    F = list(range(n_samples))

    eqp_cuts, eqp_cuts_set = [], set()

    cuts_added_per_num_removed = []

    aggregated_cuts = []

    for num_removed in range(max_removed+1):
        removed_feature_combos = itertools.combinations(F, num_removed)
        num_cuts_added = 0

        for removed_features in removed_feature_combos:
            aggregated_bound = 0
            aggregated_idx = set()
            bound_moved, _, eqp_idx, eqp_max = _find_equivalent_points(np.delete(X, removed_features, axis=1), y)

            for idx, rhs_bound in zip(eqp_idx,eqp_max):
                if idx not in eqp_cuts_set:
                    eqp_cuts.append((idx,rhs_bound,removed_features))
                    eqp_cuts_set.add(idx)
                    num_cuts_added += 1

                    if aggregate:
                        for i in idx:
                            if i in aggregated_idx:
                                print('SHOULD NOT BE HERE')
                            else:
                                aggregated_idx.add(i)
                        aggregated_bound += rhs_bound

            if len(aggregated_idx) > 0:
                aggregated_idx = list(aggregated_idx)
                aggregated_idx.sort()
                aggregated_cuts.append((tuple(aggregated_idx), aggregated_bound, removed_features))

        cuts_added_per_num_removed.append(num_cuts_added)
    if aggregate:
        return aggregated_cuts
    else:
        return eqp_cuts

def Equivalent_Points(X,y,max_removed=1,aggregate=False):
    return Equivalent_Points_Brute(X,y,max_removed=max_removed,aggregate=aggregate)

def _find_equivalent_points(X,y):
    n_samples, n_features = X.shape
    I = range(n_samples)
    F = range(n_features)

    equivalent_points = {}


    for i in I:
        sample_X = tuple(X[i, :])
        sample_y = y[i]
        if sample_X not in equivalent_points:
            equivalent_points[sample_X] = {'y': {sample_y: 0},
                                           'idx': []}

        equivalent_points[sample_X]['idx'].append(i)

        if sample_y not in equivalent_points[sample_X]['y']:
            equivalent_points[sample_X]['y'][sample_y] = 1
        else:
            equivalent_points[sample_X]['y'][sample_y] += 1

    equivalent_points_found = 0
    bound_moved = 0
    eqp_idx_all = []
    eqp_max_all = []

    for sample_X in equivalent_points:
        if len(equivalent_points[sample_X]['y']) > 1:
            eqp_idx = equivalent_points[sample_X]['idx']
            eqp_max = max(equivalent_points[sample_X]['y'].values())
            bound_moved += sum(equivalent_points[sample_X]['y'].values()) - eqp_max
            equivalent_points_found += 1
            eqp_max_all.append(eqp_max)
            eqp_idx_all.append(tuple(eqp_idx))

    return bound_moved, equivalent_points_found, eqp_idx_all, eqp_max_all


class logger(object):
    def __init__(self,console,mode=1):
        self.terminal = console
        self.mode = mode

    def SetLogFile(self,LogFile):
        self.log = open(LogFile,"a")

    def CloseLogFile(self):
        self.log.close()

    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
            pass