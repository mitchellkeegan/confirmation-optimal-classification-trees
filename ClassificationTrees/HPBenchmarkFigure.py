import os
import itertools

import pandas as pd
import matplotlib.pyplot as plt

from DataUtils import valid_datasets

name_dict = {'Quantile Buckets': 'QB-5',
             'Quantile Thresholds': 'QT-4',
             'Features Removed': 'FR',
             'Disaggregate': 'DA',
             'True': 'T',
             'False': 'F',
             'Disaggregate Alpha': 'DA',
             'EQP Basic': 'EQPB',
             'EQP Chain': 'EQPC',
             'EQP Chain Alt1': 'EQPCA',
             'Primal Heuristic Cutting Planes': 'PH Cuts'}

# Crude method to get the encoding/dataset combinations that are valid
eqp_features_removed = None
if eqp_features_removed is not None:
    eqp_filter = ([dataset + ' ' + 'None' for dataset in valid_datasets[f'eqp{eqp_features_removed}']['categorical']] +
                  [dataset + ' ' + 'Quantile Buckets' for dataset in valid_datasets[f'eqp{eqp_features_removed}']['numerical']['Quantile Buckets-5'] + valid_datasets[f'eqp{eqp_features_removed}']['mixed']['Quantile Buckets-5']] +
                  [dataset + ' ' + 'Quantile Thresholds' for dataset in valid_datasets[f'eqp{eqp_features_removed}']['numerical']['Quantile Thresholds-5'] + valid_datasets[f'eqp{eqp_features_removed}']['mixed']['Quantile Thresholds-5']])
else:
    eqp_filter = None

depth_filter = None
dataset_filter = None

# Note that bucketisation experiments have polluted some results files.
# Include filtering on "None" to capture categorical data and exclude bucketisation encoding
encoding_filter = ['None','Quantile Buckets']
# encoding_filter = ['Quantile Thresholds', 'Quantile Buckets', 'None']
# dataset_filter = valid_datasets['numerical'] + valid_datasets['mixed']

# encoding_filter = None
# dataset_filter = valid_datasets['categorical']

depth_filter = [2,3,4]

# fig_title = 'test'
fig_title = 'No Feature Reuse (Bucket Style) Benchmark'

base_model = 'BendOCT'
anchor_model = None
tablename_suffix = 'Benchmark'

base_dir = os.getcwd()

if anchor_model is not None:
    filename = ''.join(anchor_model.split()) + tablename_suffix
    file_dir = os.path.join(base_dir, 'Results', base_model, ''.join(anchor_model.split()) + tablename_suffix)
else:
    # If doing general comparison then write the table name and directory here
    filename = 'No Feature Reuse (Bucket Style) Benchmark'
    file_dir = os.path.join(base_dir, 'Results', 'Report')


FlowOCT_Model = {'Base Model': 'FlowOCT',
                 'Prefix': '',
                 'Feature Name': 'FlowOCT',
                 'Suffix': '',
                 'Hyperparameters': {}}

BendOCT_Model = {'Base Model': 'BendOCT',
                 'Prefix': '',
                 'Feature Name': 'BendOCT',
                 'Suffix': ['Single','Numerical'],
                 'Hyperparameters': {}}

SubproblemLP_Model = {'Base Model': base_model,
                      'Feature Name': 'Subproblem LP',
                      'Suffix': ['', 'Numerical'],
                      'Hyperparameters': {}}

SubproblemDual_Model = {'Base Model': base_model,
                      'Feature Name': 'Subproblem Dual',
                      'Suffix': ['', 'Numerical'],
                      'Hyperparameters': {}}

NoFeatureReuse_Model = {'Base Model': base_model,
                        'Prefix': '',
                        'Feature Name': 'No Feature Reuse',
                        'Suffix': ['Single','Numerical'],
                        'Hyperparameters': {}}

PrimalHeuristic_Model = {'Base Model': base_model,
                         'Prefix': '',
                         'Feature Name': 'Primal Heuristic',
                         'Suffix': ['Single','Numerical'],
                         'Hyperparameters': {}}

EQPBasic_Model = {'Base Model': base_model,
                  'Type': 'Initial Cut',
                  'Prefix': '',
                  'Feature Name': 'EQP Basic',
                  'Suffix': ['Single','Numerical'],
                  'Hyperparameters': {'Features Removed': [eqp_features_removed]}}

EQPChain_Model = {'Base Model': base_model,
                  'Type': 'Initial Cut',
                  'Prefix': '',
                  'Feature Name': 'EQP Chain',
                  'Suffix': ['Single','Numerical'],
                  'Hyperparameters': {'Features Removed': [eqp_features_removed],
                                      'Disaggregate Alpha': [True,False]}}

# EQPChainLazy_Model = {'Base Model': base_model,
#                       'Prefix': '',
#                       'Feature Name': 'EQP Chain',
#                       'Suffix': 'Lazy',
#                       'Hyperparameters': {'Features Removed': [eqp_features_removed],
#                                           'Lazy': [1,2,3]}}
#
# EQPChainAlt1_Model = {'Base Model': base_model,
#                       'Prefix': '',
#                       'Feature Name': 'EQP Chain Alt1',
#                       'Hyperparameters': {'EQP Chain Alt1-Features Removed': [eqp_features_removed],
#                                           'EQP Chain Alt1-Disaggregate': [True]}}

# PHCuttingPlanesSLOW_Model = {'Base Model': base_model,
#                          'Type': 'Cutting Plane',
#                          'Prefix': '',
#                          'Feature Name': 'Primal Heuristic Cutting Planes',
#                          'Suffix': 'SLOW',
#                          'Tag': 'SLOW',
#                          'Hyperparameters': {}}

PHCuttingPlanes_Model = {'Base Model': base_model,
                         'Type': 'Cutting Plane',
                         'Prefix': '',
                         'Feature Name': 'Primal Heuristic Cutting Planes',
                         'Hyperparameters': {}}

models = [BendOCT_Model,
          NoFeatureReuse_Model]

base_dir = os.getcwd()

def merge_dfs(model, prefix, suffixes, extra_tag):
    Filenames = [prefix + ''.join(model['Feature Name'].split()) + suffix + 'Benchmark' for suffix in suffixes]
    FileBases = [os.path.join(base_dir, 'Results', model['Base Model'], Filename) for Filename in Filenames]

    df_list = []

    for Filebase, Filename in zip(FileBases, Filenames):
        df_list.append(pd.read_csv(os.path.join(Filebase, Filename + '.csv')))

    return pd.concat(df_list, axis=0, join ='outer', ignore_index=True)

# For each model, load the relevant results from csv file into dataframe
# and generate sets of required combinations of hyperparameters
for model in models:
    prefix = model.get('Prefix','')
    suffix = model.get('Suffix','')
    extra_tag = model.get('Tag','')

    if isinstance(suffix,list):
        df = merge_dfs(model, prefix, suffix, extra_tag)
    else:
        model['Filename'] = prefix + ''.join(model['Feature Name'].split()) + suffix + 'Benchmark'
        model['File Base'] = os.path.join(base_dir, 'Results', model['Base Model'], model['Filename'])
        df = pd.read_csv(os.path.join(model['File Base'], model['Filename'] + '.csv'))

    # By default set encoding scheme to "None"
    if 'Encoding Scheme' in df:
        df['Encoding Scheme'] = df['Encoding Scheme'].fillna('None')
    else:
        df['Encoding Scheme'] = 'None'

    hyperparameters = model['Hyperparameters']

    if len(hyperparameters) > 0:
        feature_type = model.get('Type', 'Opt')
        hp_filters = {}
        hp_names, hp_values = hyperparameters.keys(), hyperparameters.values()
        for hp_combo in itertools.product(*hp_values):
            if feature_type in ['Initial Cut', 'Cutting Plane']:
                hp = {model['Feature Name'] + '-' + hp_name: hp_value for hp_name, hp_value in zip(hp_names, hp_combo)}
                if len(extra_tag) > 0:
                    tag_string = f' {extra_tag} -'
                else:
                    tag_string = f'-'
                hp_combo_string = (name_dict.get(model['Feature Name'],model['Feature Name']) + tag_string +
                                   ', '.join(f'{name_dict.get(hp_name, hp_name)}={name_dict.get(str(hp_value), hp_value)}'
                                             for hp_name, hp_value in zip(hp_names, hp_combo)))
            else:
                hp = {hp_name: hp_value for hp_name, hp_value in zip(hp_names, hp_combo)}
                hp_combo_string = ', '.join(f'{name_dict.get(hp_name, hp_name)}={name_dict.get(str(hp_value), hp_value)}'
                                            for hp_name, hp_value in zip(hp_names, hp_combo))
            hp_filters[hp_combo_string] = hp
        model['hp Filters'] = hp_filters
    else:
        feature_name = name_dict.get(model['Feature Name'],model['Feature Name'])
        if 'Tag' in model:
            model['hp Filters'] = {feature_name + f' ({model['Tag']})': {}}
        else:
            model['hp Filters'] = {feature_name: {}}

    # df = filter_df(df,dataset_filter, depth_filter, encoding_filter)

    # Filter for the valid combinations of as per the eqp-FR setting
    if eqp_filter is not None:
        df = df[(df['Dataset'] + ' ' + df['Encoding Scheme']).isin(eqp_filter)]

    df['Instance Identifier'] = df['Dataset'] + ' ' + df['Encoding Scheme'] + ' ' + df['depth'].astype('str')
    model['Instance Combos'] = set(df['Instance Identifier'])

    model['df'] = df

for model in models:
    print(model['hp Filters'])

# Filter the hyperparameters and benchmarks based on what dataset + depth combinations are actually available
valid_combos = set.intersection(*[model['Instance Combos'] for model in models])

# TODO: Reimplement with group_by?
for model in models:
    model['df'] = model['df'].loc[model['df']['Instance Identifier'].isin(valid_combos)]
    model['df'].reset_index(inplace=True)


def plot_filtered_rows(rows, axs, name, max_opt_solve_time, max_opt_gap):
    rows_time = rows.loc[rows['Model Status'] == 2].sort_values(by='Solve Time', ignore_index=True)
    rows_gap = rows.loc[rows['Model Status'] == 9].sort_values(by='Gap', ignore_index=True)

    if rows_time['Solve Time'].max() > max_opt_solve_time:
        max_opt_solve_time = rows_time['Solve Time'].max()

    if rows_gap['Gap'].max() > max_opt_gap:
        max_opt_gap = rows_gap['Gap'].max()

    axs[0].plot(rows_time['Solve Time'], rows_time.index, label=name, drawstyle='steps')
    axs[1].plot(rows_gap['Gap'] * 100, rows_gap.index + rows_time.shape[0], label=name, drawstyle='steps')

    return max_opt_solve_time, max_opt_gap

max_opt_solve_time = 0
max_opt_gap = 0

num_rows = None

# Loop over the models and generate rows for each hyperparameter combo
for model in models:
    hp_filters = model['hp Filters']
    df = model['df']

    model['Rows'] = {}

    for name, filter in hp_filters.items():
        b = pd.Series([True] * df.shape[0])
        for column, condition in filter.items():
            b &= (df[column] == condition)

        if dataset_filter is not None:
            b &= (df['Dataset'].isin(dataset_filter))
        if depth_filter is not None:
            b &= (df['depth'].isin(depth_filter))

        if encoding_filter is not None:
            if 'Encoding Scheme' in df.columns:
                b &= (df['Encoding Scheme'].isin(encoding_filter))
            else:
                print('Encoding Filter specified but dataset does not have an encoding scheme (probably categorical data)')

        # Filter rows
        rows = df.loc[b]

        if num_rows is not None:
            if len(rows) != num_rows:
                print('Row ' + name + 'size does not agree with previous rows')
        else:
            num_rows = len(rows)

        rows_time = rows.loc[rows['Model Status'] == 2].sort_values(by='Solve Time', ignore_index=True)
        rows_gap = rows.loc[rows['Model Status'] == 9].sort_values(by='Gap', ignore_index=True)

        if rows_time['Solve Time'].max() > max_opt_solve_time:
            max_opt_solve_time = rows_time['Solve Time'].max()

        if rows_gap['Gap'].max() > max_opt_gap:
            max_opt_gap = rows_gap['Gap'].max()

        model['Rows'][name] = {'Time': rows_time,
                               'Gap': rows_gap}

fig, axs = plt.subplots(1, 2, sharey=True)
fig.subplots_adjust(wspace=0)

# Loop over each model and it's set of rows and plot on figure
for model in models:
    for name, rows_dict in model['Rows'].items():
        rows_time = rows_dict['Time']
        rows_gap = rows_dict['Gap']

        axs[0].plot(rows_time['Solve Time'], rows_time.index, label=name, drawstyle='steps')
        axs[1].plot(rows_gap['Gap'] * 100, rows_gap.index + rows_time.shape[0], label=name, drawstyle='steps')


axs[0].set_xlabel('Solve Time (s)')
axs[1].set_xlabel('Optimality Gap (%)')
axs[0].set_xlim(0.0, max_opt_solve_time)
axs[1].set_xlim(0.0, max_opt_gap * 100)
axs[0].set_ylabel('Number of Instances')
axs[0].set_ylim(0,num_rows+1)
plt.legend()
plt.suptitle(fig_title)
plt.savefig(os.path.join(file_dir, filename + '.png'))
plt.show()