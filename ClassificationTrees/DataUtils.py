import os
import pickle

import pandas as pd
import numpy as np


categorical_datasets = ['balance-scale', 'breast-cancer', 'car_evaluation', 'hayes-roth', 'house-votes-84', 'kr-vs-kp',
                        'monk1', 'monk2', 'monk3', 'soybean-small', 'spect', 'tic-tac-toe']

numerical_datasets = ['iris','wine', 'plrx', 'wpbc', 'parkinsons','sonar','wdbc',
                      'banknote', 'ILPD', 'transfusion', 'biodeg', 'ozone-one', 'segmentation',
                      'spambase']

mixed_datasets = ['hepatitis', 'fertility','ionosphere', 'thoracic', 'credit', 'seismic-bumps', 'ann-thyroid']

compressible_datasets = ['breast-cancer', 'car_evaluation', 'hayes-roth', 'house-votes-84', 'spect',
                         'iris', 'plrx', 'ILPD', 'transfusion', 'biodeg', 'banknote', 'segmentation', 'spambase',
                         'thoracic', 'seismic-bumps', 'ann-thyroid']

SplitOCT_datasets = ['breast-cancer', 'hayes-roth', 'house-votes-84', 'monk1', 'monk2', 'monk3', 'soybean-small', 'spect']

eqp0_datasets = {'categorical': ['breast-cancer', 'car_evaluation', 'hayes-roth', 'spect'],
                 'numerical': {'Quantile Buckets-5': ['iris', 'banknote','ILPD', 'transfusion', 'biodeg', 'segmentation', 'spambase'],
                               'Quantile Thresholds-5': ['iris', 'banknote','ILPD', 'transfusion', 'biodeg', 'segmentation', 'spambase']},
                 'mixed': {'Quantile Buckets-5': ['fertility', 'thoracic', 'credit', 'seismic-bumps', 'ann-thyroid'],
                           'Quantile Thresholds-5': ['fertility', 'thoracic', 'seismic-bumps', 'ann-thyroid']}}

eqp1_datasets = {'categorical': ['breast-cancer', 'car_evaluation', 'house-votes-84', 'kr-vs-kp', 'monk2', 'spect'],
                 'numerical': {'Quantile Buckets-5': [],
                               'Quantile Thresholds-5': ['iris', 'banknote','ILPD', 'transfusion', 'biodeg', 'segmentation', 'spambase']},
                 'mixed': {'Quantile Buckets-5': ['thoracic', 'seismic-bumps', 'ann-thyroid'],
                           'Quantile Thresholds-5': ['thoracic', 'credit', 'seismic-bumps', 'ann-thyroid']}}

eqp2_datasets = {'categorical': ['balance-scale', 'breast-cancer', 'car_evaluation', 'hayes-roth', 'house-votes-84',
                                 'kr-vs-kp', 'monk1', 'monk2', 'monk3', 'spect'],
                 'numerical': {'Quantile Buckets-5': ['iris', 'banknote', 'ILPD', 'transfusion', 'biodeg', 'segmentation', 'spambase'],
                               'Quantile Thresholds-5': ['iris', 'banknote','ILPD', 'transfusion', 'biodeg', 'segmentation', 'spambase']},
                 'mixed': {'Quantile Buckets-5': ['fertility', 'thoracic', 'credit', 'seismic-bumps', 'ann-thyroid'],
                           'Quantile Thresholds-5': ['fertility', 'thoracic', 'credit', 'seismic-bumps', 'ann-thyroid']}}


# eqp0_datasets = ['breast-cancer', 'car_evaluation', 'hayes-roth', 'spect',
#                  'iris', 'banknote','ILPD', 'transfusion', 'biodeg', 'segmentation', 'spambase',
#                  'fertility', 'thoracic', 'credit', 'seismic-bumps', 'ann-thyroid']
#
# eqp1_datasets = ['breast-cancer', 'car_evaluation', 'house-votes-84', 'kr-vs-kp', 'monk2', 'spect',
#                  'thoracic', 'seismic-bumps', 'ann-thyroid']

# eqp2_datasets = ['balance-scale', 'breast-cancer', 'car_evaluation', 'hayes-roth', 'house-votes-84', 'kr-vs-kp', 'monk1', 'monk2', 'monk3', 'spect',
#                  'iris', 'banknote', 'ILPD', 'transfusion', 'biodeg', 'segmentation', 'spambase',
#                  'fertility', 'thoracic', 'credit', 'seismic-bumps', 'ann-thyroid']


all_datasets = categorical_datasets + numerical_datasets + mixed_datasets

valid_datasets = {'all': all_datasets,
                  'categorical': categorical_datasets,
                  'numerical': numerical_datasets,
                  'mixed': mixed_datasets,
                  'compressible': compressible_datasets,
                  'eqp0': eqp0_datasets,
                  'eqp1': eqp1_datasets,
                  'eqp2': eqp2_datasets}

def extract_category_info(columns, instance):
    # if instance not in ['house-votes-84','spect']:

    features = [s.rsplit('.')[0] for s in columns]

    curr_feat = features[0]
    groupings = [[]]

    for f, Cat_feature in enumerate(features):
        if Cat_feature == curr_feat:
            groupings[-1].append(f)
        else:
            curr_feat = Cat_feature
            groupings.append([f])

    return groupings

def categorical_encoding(feature_series,feat_name):
    unique_levels = feature_series.unique()
    unique_levels.sort()

    new_series = []

    if len(unique_levels) == 1:
        # Ignore features with only one level
        return None
    elif len(unique_levels) == 2:
        # Encode features with two levels as a single binary variable
        # By default assume that the second feature (in the sorted order) corresponds to one
        level = unique_levels[1]
        series = (feature_series == level).astype('int8')
        series.name = feat_name + f'.{level}'
        new_series.append(series)
    else:
        # Otherwise onehot encode as normal
        for level in unique_levels:
            series = (feature_series == level).astype('int8')
            series.name = feat_name + f'.{level}'
            new_series.append(series)

    return pd.concat(new_series,axis=1)

def bucket_encoding(feature_series, encoding_scheme, num_buckets, feat_name):

    # feature_df_sorted = feature_df.sort_values(by='feature')
    #
    # feature_vals, targets = feature_df_sorted['feature'].to_numpy(), feature_df_sorted['target'].to_numpy()

    # num_samples = len(feature_vals)
    edges = []

    if encoding_scheme == 'Quantile Buckets':
        categories, bins = pd.qcut(feature_series, num_buckets ,retbins=True, duplicates='drop')
        bin_df = pd.get_dummies(categories).astype('int8')
        bin_df.columns = [feat_name + f'.buck{i}' for i in range(bin_df.shape[1])]
        return bin_df, bins
    elif encoding_scheme == 'Equal Buckets':
        return None, None
    else:
        return None, None

def threshold_encoding(feature_df, encoding_scheme, num_buckets, feat_name):
    EPS = 1e-8
    df = feature_df.sort_values(by='feature')

    feature_vals = df['feature']

    # feature_vals, targets = feature_df_sorted['feature'].to_numpy(), feature_df_sorted['target'].to_numpy()

    num_samples = df.shape[0]
    thresholds = []

    saved_targets = set()

    index_map = df.index

    if encoding_scheme in ['Full', 'Bucketisation']:
        for i in range(num_samples - 1):
            # Only want to introduce a threshold if difference between adjacent feature values are above some tolerance
            f_i = df.loc[index_map[i], 'feature']
            f_ii = df.loc[index_map[i + 1], 'feature']
            if f_ii - f_i > EPS:
                if encoding_scheme == 'Full':
                    thresholds.append((f_ii + f_i) / 2)
                elif (encoding_scheme == 'Bucketisation'):
                    # When treating ordinal variables like numerical variables we want to check if some sample with the same feature
                    # value differs in the target, not only the "last" sample in our sample ordering
                    t_i = df.loc[index_map[i], 'target']
                    t_ii = df.loc[index_map[i+1], 'target']
                    saved_targets.add(t_i)
                    for targ in saved_targets:
                        if targ != t_ii:
                            thresholds.append((f_ii + f_i) / 2)
                            continue
                    saved_targets = set()
            else:
                saved_targets.add(df.loc[index_map[i], 'target'])

    elif encoding_scheme == 'Quantile Thresholds':
        num_unique_values = len(feature_vals.unique())
        quantiles = np.linspace(0,1,num_buckets+1)[1:-1]
        thresholds = (feature_vals.quantile(quantiles, interpolation='midpoint').unique() + 1e-8).tolist()

    print(thresholds)

    bin_feature_series = []
    for i, threshold in enumerate(thresholds):
        series = (feature_df['feature'] >= threshold).astype('int8')
        series.name = feat_name + f'.th{i}'
        bin_feature_series.append(series)

    return pd.concat(bin_feature_series, axis=1), thresholds

def numerical_encoding(feature_df, encoding_scheme, feat_name, num_buckets=None):
    if encoding_scheme in ['Quantile Buckets', 'Equal Buckets']:
        return bucket_encoding(feature_df['feature'], encoding_scheme, num_buckets, feat_name)
    elif encoding_scheme in ['Full', 'Bucketisation', 'Quantile Thresholds']:
        return threshold_encoding(feature_df, encoding_scheme, num_buckets, feat_name)

def encode_dataset(raw_data,dataset_name,encoding_scheme=None,num_buckets=None):

    force_numerical = {'hepatitis': [0,14,15,17],
                       'wpbc': [31],
                       'wine': [4,12],
                       'thoracic': [15],
                       'ILPD': [0,4,5,6],
                       'credit': [13,14],
                       'transfusion': [0,1,2,3],
                       'biodeg': [2,4,5,6,8,9,10,15,22,31,32,33,34,37,40],
                       'ozone-one': [66,69,70],
                       'segmentation': [0,1],
                       'seismic-bumps': [3,4,5,6,8,9,10,11,13,14,15,16,17],
                       'spambase': [55,56]}

    force_categorical = {'fertility': [0],
                         'seismic-bumps': [12]}

    # If target class is not in the last row, need to specify position. NOTE: THIS IS AFTER ANY ROWS ARE DROPPED
    target_col_dict = {'balance-scale':0,
                       'breast-cancer':0,
                       'house-votes-84': 0,
                       'spect': 0,
                       'hepatitis': 0,
                       'wine': 0,
                       'wpbc': 0,
                       'parkinsons': 16,
                       'wdbc': 0,
                       'segmentation': 0}

    target_col_idx = target_col_dict.get(dataset_name,raw_data.shape[1] - 1)

    target_series = raw_data.iloc[:,target_col_idx]
    target_series.name = 'target'

    unique_targets = target_series.unique()
    target2enc = {targ: i for i,targ in enumerate(unique_targets)}
    enc2targ = {i: targ for i,targ in enumerate(unique_targets)}

    target_series = target_series.replace(to_replace=target2enc)

    all_features_df = raw_data.drop(target_col_idx, axis=1)
    num_samples, num_features = all_features_df.shape

    encoded_dfs = []
    num_cat_features = 0
    num_numerical_features = 0

    # Set up info which maps the categorical/numerical features to the related binary features
    aux_info = {}
    all2encMap = []
    cat2binMap = []
    num2binMap = []
    feature_edges = []

    num_bin_features = 0

    for f in range(num_features):
        feature_series = all_features_df.iloc[:,f]
        feature_series.name = 'feature'

        # Rough way to check if a feature is categorical or numerical
        if (pd.api.types.is_float_dtype(feature_series.dtype) and f not in force_categorical.get(dataset_name,set())) or (f in force_numerical.get(dataset_name,set())):
            assert (encoding_scheme is not None)

            # Assert that the dtype must be numeric. Allowed to be integer which can happen if user adds feature to force_numeric
            assert pd.api.types.is_numeric_dtype(feature_series.dtype)

            unique_features = feature_series.unique()
            if len(unique_features) == 2:
                print(f'Feature {f} in dataset {dataset_name} is numeric but only has {len(feature_series.unique())} unique features')
                print('Recommend adding dataset/feature to force_categorical')
            if len(unique_features) == 1:
                # If only one unique value then we should not use the feature
                all2encMap.append('Unencoded')
                continue

            feature_bin_df, edges = numerical_encoding(pd.concat([feature_series,target_series],axis=1),
                                                      encoding_scheme,
                                                      f'Feat{f}_Num{num_cat_features}',
                                                      num_buckets=num_buckets)

            if feature_bin_df is None:
                all2encMap.append('Unencoded')
                continue

            feature_edges.append(edges)

            all2encMap.append(f'Num{num_numerical_features}')

            encoded_dfs.append(feature_bin_df)
            new_bin_features = feature_bin_df.shape[1]

            # Keep track of what binary features each categorical feature maps to
            num2binMap.append(range(num_bin_features, num_bin_features + new_bin_features))
            num_bin_features += new_bin_features

            num_numerical_features += 1

        else:
            unique_features = feature_series.unique()

            # If this does not hold it is likely that the feature is continuous or ordinal
            if pd.api.types.is_integer_dtype(feature_series.dtype) and (len(unique_features) > min(num_samples // 10, 5)):
                if f not in force_categorical.get(dataset_name,set()):
                    print(f'Feature {f} in dataset {dataset_name} has a non-float feature with {len(unique_features)} unique features and {num_samples} datapoints')
                    print('Recommend adding dataset/feature to force_numerical or force_categorical')

            feature_bin_df = categorical_encoding(feature_series,f'Feat{f}_Cat{num_cat_features}')
            if feature_bin_df is None:
                all2encMap.append('Unencoded')
                continue

            all2encMap.append(f'Cat{num_cat_features}')

            encoded_dfs.append(feature_bin_df)
            new_bin_features = feature_bin_df.shape[1]

            # Keep track of what binary features each categorical feature maps to
            cat2binMap.append(range(num_bin_features,num_bin_features + new_bin_features))
            num_bin_features += new_bin_features

            num_cat_features += 1

    if len(feature_edges) > 0:
        if encoding_scheme in ['Quantile Buckets', 'Equal Buckets']:
            aux_info['Buckets'] = feature_edges
        elif encoding_scheme in ['Full', 'Bucketisation']:
            aux_info['Thresholds'] = feature_edges


    aux_info['Categorical Feature Map'] = cat2binMap
    aux_info['Numerical Feature Map'] = num2binMap
    aux_info['Original Feature Map'] = all2encMap
    aux_info['Target to Encoded Map'] = target2enc

    encoded_dfs.append(target_series)

    return pd.concat(encoded_dfs,axis=1), aux_info

def load_raw_data(dataset):
    file_extensions = ['.data','.csv','.train','.test', '.txt', '.all-data', '.arff', '.dat']

    na_values = ['?']

    dropped_cols_dict = {'hayes-roth': 0,
                         'wpbc': [0,2],
                         'parkinsons': 0,
                         'bands': 0,
                         'ionosphere': 1,
                         'wdbc': 0,
                         'ozone-one': 0}

    # By default use ',' as a seperator but must accommodate datasets with different seperators
    sep_dict = {'car_evaluation': ';',
                'monk1': ';',
                'monk2': ';',
                'monk3': ';',
                'plrx': '\t ',
                'seeds': '\t',
                'biodeg': ';',
                'ann-thyroid': '\s+'}

    sep = sep_dict.get(dataset, ',')

    needs_header = ['car_evaluation', 'monk1', 'monk2', 'monk3', 'parkinsons', 'transfusion']
    header = 0 if dataset in needs_header else None

    df_list = []

    for ext in file_extensions:
        dataset_file = os.path.join('Datasets',dataset + ext)

        if (ext == '.test') and dataset == 'hayes-roth':
            # Don't want the hayes-roth test data
            continue

        try:
            new_data = pd.read_csv(dataset_file,header=header,na_values=na_values,sep=sep).dropna()

            # If the data came with a header, forcibly overwrite it with the columns indices to allow us to locate the target column by index only
            new_data.columns = [i for i in range(new_data.shape[1])]

            if dataset in dropped_cols_dict:
                dropped_cols = dropped_cols_dict[dataset]
                new_data = new_data.drop(dropped_cols,axis=1)
                new_data.columns = [i for i in range(new_data.shape[1])]

            # Since pandas can only represent NaN as float types, any integer columns with NaN in them will be cast to a float and remain so after dropna() is applied
            # convert_dtypes() allows us to infer that these should be integer types, but it uses the Nullable Int64 type (as opposed to int64) so we must
            # manually convert these back to normal numpy non-nullable types for compatibility with later code
            # To my knowledge Pandas doesn't have an option with read_csv to do this easily
            for col in new_data.columns:
                if pd.api.types.is_float_dtype(new_data[col]):
                    new_data[col] = new_data[col].convert_dtypes()
                    if pd.api.types.is_integer_dtype(new_data[col]):
                        new_data[col] = new_data[col].astype('int64')
                    if pd.api.types.is_float_dtype(new_data[col]):
                        new_data[col] = new_data[col].astype('float64')

            print(f'Found {dataset} raw data in {dataset + ext}')
            df_list.append(new_data)

        except:
            pass

    # If we have found raw data in our files, concatenate the data found and drop rows with NaN
    # Does not check if concatenated data has the same number of columns so it will error out if this is not the case
    if len(df_list) > 0:
        return pd.concat(df_list,axis=0,ignore_index=True)
    else:
        return None

def load_instance(dataset,
                  compress=False,
                  encoding_scheme=None,
                  num_buckets=None,
                  force_encoding=False):

    valid_encoding_schemes = ['Full', 'Bucketisation', 'Quantile Thresholds', 'Quantile Buckets', 'Equal Buckets']

    # Check that given dataset and settings are valid. If so create a filename for the encoded data.
    if dataset in categorical_datasets:
        dataset_filename = dataset + '_enc'
    elif dataset in numerical_datasets or dataset in mixed_datasets:
        if encoding_scheme is not None:
            if encoding_scheme in valid_encoding_schemes:
                dataset_filename = dataset + '_' + encoding_scheme.replace(' ','_')
                if encoding_scheme in ['Quantile Thresholds', 'Quantile Buckets', 'Equal Buckets']:
                    if isinstance(num_buckets,int):
                        dataset_filename += f'_{num_buckets}'
                    else:
                        print(f'num_buckets={num_buckets} invalid for {encoding_scheme} encoding scheme. Please choose an integer')
                    dataset_filename += '_enc'
            else:
                print(f'{encoding_scheme} is not a known encoding scheme. Please choose one from {valid_encoding_schemes}')
                return None
        else:
            print(f'Need to specify an encoding scheme for dataset {dataset}')
            return None
    else:
        print(f'{dataset} dataset not found in set of valid datasets')
        return None

    # If user has not requested a re-encoding of the raw data, attempt to load encoded data from file
    # If this fails for any reason, set the force_encoding flag to True
    if not force_encoding:
        try:
            dataset_file = os.path.join('Datasets', dataset_filename + '.csv')
            dataset_aux_file = os.path.join('Datasets', 'AuxFiles', dataset_filename + '.pickle')

            data = pd.read_csv(dataset_file)
            with open(dataset_aux_file,'rb') as f:
                aux_info = pickle.load(f)

        except FileNotFoundError:
            print('Could not find encoded dataset files. Attempting to re-encode from raw data')
            force_encoding = True

        except:
            print(f'Something went wrong loading in dataset {dataset}')
            return None

    # If encoding from raw data requested by user or saved encoding could not be found
    # attempt to load in raw data and pass to encode_dataset
    if force_encoding:
        raw_data = load_raw_data(dataset)
        if raw_data is not None:
            data, aux_info = encode_dataset(raw_data,dataset,encoding_scheme=encoding_scheme,num_buckets=num_buckets)
            if data is not None and aux_info is not None:
                dataset_file = os.path.join('Datasets', dataset_filename + '.csv')
                dataset_aux_file = os.path.join('Datasets', 'AuxFiles', dataset_filename + '.pickle')

                data.to_csv(dataset_file,index=False)
                with open(dataset_aux_file,'wb') as f:
                    pickle.dump(aux_info,f)
                pass
            else:
                print(f'Issue encountered when encoding {dataset} data')

        else:
            print(f'Could not find raw data file for {dataset} dataset')

    data = data.to_numpy(dtype='int')

    print('\n' + '-' * 5 + f'Loaded in {dataset} Dataset' + '-' * 5)

    n_samples, _ = data.shape
    I = range(n_samples)

    # TODO: Is there a better method than carrying around the uncompressed data just for the heuristic?

    if compress:
        sample_to_new_idx = {}
        j = 0
        idx_new = []  # List of idx (in original dataset) that remain after compression
        weights = []  # Weights for objective functions
        idx_map = []  # Maps from idx of compressed data to idx of full dataset, elements are lists
        idx_map_inverse = []  # Maps from full dataset idx to compressed idx, elements are ints

        for i in I:
            sample = tuple(data[i, :])
            if sample not in sample_to_new_idx:
                sample_to_new_idx[sample] = j
                weights.append(1)
                idx_map.append([i])
                idx_map_inverse.append(j)
                idx_new.append(i)
                j += 1
            else:
                jj = sample_to_new_idx[sample]
                weights[jj] += 1
                idx_map[jj].append(i)
                idx_map_inverse.append(jj)

        if n_samples > len(sample_to_new_idx):
            print(f'Compressed {n_samples} datapoints into {len(sample_to_new_idx)} datapoints')
        else:
            print(f'Dataset has no equal datapoints (cannot be compressed)')

    else:
        idx_new = [i for i in I]
        weights = [1 for _ in I]
        idx_map = {j: [j] for j in range(n_samples)}
        idx_map_inverse = idx_map

    # Dataset with no compression applied (needed for CART Heuristic)
    Xf, yf = data[:, :-1], data[:, -1]

    # Dataset with compression applied (X=Xf and y=yf if no compression)
    X, y = Xf[idx_new, :], yf[idx_new]

    I, F, K, info = Dataset_Info(X, y, aux_info)
    print(info)

    instance_data = {'X': X,
                     'y': y,
                     'Xf': Xf,
                     'yf': yf,
                     'I': I,
                     'F': F,
                     'K': K,
                     'weights': weights,
                     'idxc_to_idxf': idx_map,
                     'idxf_to_idxc': idx_map_inverse,
                     'compressed': compress,
                     'name': dataset,
                     'encoding': encoding_scheme}

    if encoding_scheme in [None,'Bucketisation', 'Full']:
        instance_data['encoding'] = encoding_scheme
    elif encoding_scheme == 'Quantile Buckets':
        instance_data['encoding'] = f'QB_{num_buckets}'
    elif encoding_scheme == 'Equal Buckets':
        instance_data['encoding'] = f'EB_{num_buckets}'

    instance_data |= aux_info

    return instance_data

def Dataset_Info(X,y,aux_info):

    n_samples, n_features = X.shape
    I = range(n_samples)
    F = range(n_features)
    K = np.unique(y).tolist()

    C = aux_info['Categorical Feature Map']
    N = aux_info['Numerical Feature Map']


    info = ''

    info += f'|I| = {len(I)}, |Fc| = {len(C)}, |Fn| = {len(N)}, |Fb| = {len(F)}, |K| = {len(K)}\n'

    class_distribution = [sum(y[i] == k for i in I)/n_samples for k in K]

    info += 'Class Distribution: ' + ','.join(f'{cd:.2f}' for cd in class_distribution) + '\n'

    num_ones_per_feature = [sum(X[i,f] > 0.5 for i in I)/n_samples for f in F]

    feature_sparsity = [max(num_ones,1-num_ones) for num_ones in num_ones_per_feature]

    info += 'Feature Sparsity: ' + ','.join(f'{fs:.2f}' for fs in feature_sparsity) + '\n'
    info += f'Average Sparsity: {sum(feature_sparsity)/len(feature_sparsity):.2f}\n'

    return I, F, K, info

