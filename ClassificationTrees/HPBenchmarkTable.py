import os
import itertools

import pandas as pd

from ClassificationTrees.DataUtils import valid_datasets

pd.options.mode.copy_on_write = True

name_dict = {'Features Removed': 'FR',
             'Disaggregate': 'DA',
             'True': 'T',
             'False': 'F',
             'Disaggregate Alpha': 'DA',
             'EQP Basic': 'EQPB',
             'EQP Chain': 'EQPC',
             'EQP Chain Alt1': 'EQPCA',
             'No Feature Reuse': 'NFR',
             'Subproblem LP': 'Sub LP',
             'Subproblem Dual': 'Sub Dual',
             'Primal Heuristic Cutting Planes': 'PH Cuts'}

base_model = 'BendOCT'
anchor_model = None
tablename_suffix = 'Benchmark'

allowed_datasets = None
allowed_encodings = None
allowed_depths = None

# allowed_datasets = valid_datasets['categorical']
allowed_datasets = [d for d in valid_datasets['numerical'] + valid_datasets['mixed']
                    if d not in {'hepatitis', 'ionosphere', 'wine', 'plrx', 'wpbc', 'parkinsons', 'sonar', 'wdbc', 'ozone-one'}]
allowed_encodings = ['Quantile Thresholds']
allowed_depths = [2,3,4]

base_dir = os.getcwd()

if anchor_model is not None:
    tablename = ''.join(anchor_model.split()) + tablename_suffix
    table_dir = os.path.join(base_dir, 'Results', base_model, ''.join(anchor_model.split()) + tablename_suffix)
else:
    # If doing general comparison then write the table name and directory here
    tablename = f'EQP Cuts Numerical (Thresholds) Benchmark'
    table_dir = os.path.join(base_dir, 'Results', 'Report')

FlowOCT_Model = {'Base Model': 'FlowOCT',
                 'Prefix': '',
                 'Feature Name': 'FlowOCT',
                 'Suffix': '',
                 'Hyperparameters': {}}

BendOCT_Model = {'Base Model': 'BendOCT',
                 'Prefix': '',
                 'Feature Name': 'BendOCT',
                 'Suffix': 'Numerical',
                 'Hyperparameters': {}}

PrimalHeuristic_Model = {'Base Model': base_model,
                         'Prefix': '',
                         'Feature Name': 'Primal Heuristic',
                         'Suffix': 'Single',
                         'Hyperparameters': {}}

EQPBasic_Model = {'Base Model': base_model,
                  'Type': 'Initial Cut',
                  'Prefix': '',
                  'Feature Name': 'EQP Basic',
                  'Suffix': 'Numerical',
                  'Hyperparameters': {'Features Removed': [0,1,2]}}

EQPChain_Model = {'Base Model': base_model,
                  'Type': 'Initial Cut',
                  'Prefix': '',
                  'Feature Name': 'EQP Chain',
                  'Suffix': 'Numerical',
                  'Hyperparameters': {'Features Removed': [1,2],
                                      'Disaggregate Alpha': [False,True]}}

EQPChainAlt1_Model = {'Base Model': base_model,
                      'Prefix': '',
                      'Feature Name': 'EQP Chain Alt1',
                      'Suffix': '',
                      'Hyperparameters': {'Features Removed': [1,2],
                                          'Disaggregate': [True]}}

NoFeatureReuse_Model = {'Base Model': base_model,
                        'Prefix': '',
                        'Feature Name': 'No Feature Reuse',
                        'Suffix': 'Numerical',
                        'Hyperparameters': {}}

SubproblemLP_Model = {'Base Model': base_model,
                        'Prefix': '',
                        'Feature Name': 'Subproblem LP',
                        'Suffix': 'Numerical',
                        'Hyperparameters': {}}

SubproblemDualInspection_Model = {'Base Model': base_model,
                                  'Prefix': '',
                                  'Feature Name': 'Subproblem Dual',
                                  'Suffix': 'Numerical',
                                  'Hyperparameters': {}}

PHCuttingPlanesSLOW_Model = {'Base Model': base_model,
                         'Type': 'Cutting Plane',
                         'Prefix': '',
                         'Feature Name': 'Primal Heuristic Cutting Planes',
                         'Suffix': 'SLOW',
                         'Tag': 'SLOW',
                         'Hyperparameters': {}}

PHCuttingPlanes_Model = {'Base Model': base_model,
                         'Type': 'Cutting Plane',
                         'Prefix': '',
                         'Feature Name': 'Primal Heuristic Cutting Planes',
                         'Hyperparameters': {}}

models = [BendOCT_Model,
          EQPBasic_Model,
          EQPChain_Model]

# For each model, load the relevant results from csv file into dataframe
# and generate sets of required combinations of hyperparameters
for model in models:
    prefix = model.get('Prefix', '')
    suffix = model.get('Suffix', '')
    extra_tag = model.get('Tag', '')
    model['Filename'] = prefix + ''.join(model['Feature Name'].split()) + suffix + 'Benchmark'
    model['File Base'] = os.path.join(base_dir, 'Results', model['Base Model'], model['Filename'])
    df = pd.read_csv(os.path.join(model['File Base'], model['Filename'] + '.csv'))

    hyperparameters = model['Hyperparameters']

    if len(hyperparameters) > 0:
        feature_type = model.get('Type', 'Opt')
        hp_filters = {}
        hp_names, hp_values = hyperparameters.keys(), hyperparameters.values()
        for hp_combo in itertools.product(*hp_values):
            if feature_type in ['Initial Cut', 'Cutting Plane']:
                hp = {model['Feature Name'] + '-' + hp_name: hp_value for hp_name, hp_value in zip(hp_names, hp_combo)}
                hp_combo_string = ', '.join(f'{name_dict.get(hp_name,hp_name)}={name_dict.get(str(hp_value),hp_value)}'
                                           for hp_name, hp_value in zip(hp_names, hp_combo))
            else:
                hp = {hp_name: hp_value for hp_name, hp_value in zip(hp_names, hp_combo)}
                hp_combo_string = ', '.join(f'{name_dict.get(hp_name, hp_name)}={name_dict.get(str(hp_value), hp_value)}'
                                            for hp_name, hp_value in zip(hp_names, hp_combo))
            hp_filters[hp_combo_string] = hp
        model['hp Filters'] = hp_filters
    else:
        feature_name = name_dict.get(model['Feature Name'], model['Feature Name'])
        if 'Tag' in model:
            model['hp Filters'] = {feature_name + f' ({model['Tag']})': {}}
        else:
            model['hp Filters'] = {feature_name: {}}

    df['Dataset and Depth'] = df['Dataset'] + ' ' + df['depth'].astype('str')
    model['Dataset and Depth Combos'] = set(df['Dataset and Depth'])

    model['df'] = df

for model in models:
    print(model['hp Filters'])


if allowed_depths is None:
    allowed_depths = df['depth'].unique()
if allowed_datasets is None:
    allowed_datasets = df['Dataset'].unique()

allowed_depths.sort()

def split_df_by_filter(df,hp_filter):
    b = pd.Series([True] * df.shape[0])
    for column, condition in hp_filter.items():
        b &= (df[column] == condition)
    return df[b]

for model in models:
    hyperparameters = model['Hyperparameters']
    df = model['df']

    # if len(hyperparameters) > 0:
    model['Filtered dfs'] = [(name, filt, split_df_by_filter(df, filt)) for name, filt in model['hp Filters'].items()]
    # else:
    #     model['Filtered dfs'] = [(model['Feature Name'], {}, df)]

def parenthesise(text):
    return '{' + f'{text}' + '}'

def multirow(depth,text):
    return '\\multirow' + parenthesise(depth) + parenthesise('*') + parenthesise(text)

def multicolumn(length,format_string,text):
    line = '\\multicolumn' + parenthesise(length) + parenthesise(format_string) + parenthesise(text)
    return line

def preamble(num_hp_combos):
    lines = '\\begin' + parenthesise('table') + '[h]\n'
    lines += '\\centering\n'
    format_string = '|c||' + 'c'*num_hp_combos + '|\n'
    lines += '\\begin' + parenthesise('tabular') + parenthesise(format_string) + '\n'
    lines += '\\hline\n'
    return lines

def hyperparameter_rows(models):
    split_col_names_over_rows = max(len(model['Filtered dfs']) for model in models) > 1

    if split_col_names_over_rows:
        line1 = multirow(2,'Dataset')
        line2 = ''
    else:
        line = 'Dataset'

    for model in models:
        feature_name = name_dict.get(model['Feature Name'], model['Feature Name'])
        if 'Tag' in model:
            feature_name += f' ({model['Tag']})'
        filtered_dfs = model['Filtered dfs']

        if len(filtered_dfs) > 1:
            line1 += '&' + multicolumn(len(filtered_dfs), 'c', feature_name)
            for name, _, _ in filtered_dfs:
                line2 += ' & ' + name

        else:

            if split_col_names_over_rows:
                line1 += ' & ' + multirow(2,feature_name)
                line2 += ' & '
            else:
                line += ' & ' + feature_name

    if split_col_names_over_rows:
        line1 += '\\\\ \n'
        line2 += '\\\\ \n \\hline \n'

        return line1 + line2
    else:
        line += '\\\\ \n \\hline \n'
        return line

def postamble(filename):
    lines = '\\hline\n'
    lines += '\\end' + parenthesise('tabular') + '\n'
    lines += '\\caption' + parenthesise(filename) + '\n'
    lines += '\\label' + parenthesise('tab: ' + filename) + '\n'
    lines += '\\end' + parenthesise('table')
    return lines

with open(os.path.join(table_dir, tablename + '.tex'),'w') as f:
    num_model_cols = sum(len(model['Filtered dfs']) for model in models)
    f.write(preamble(num_model_cols))
    # columns = ' & ' + model + ' & ' + '&'.join([f'hp{idx+1}' for idx in range(len(hp_idx_to_filter))]) + '\\\\\n'
    # f.write(columns)
    f.write(hyperparameter_rows(models))
    for depth in allowed_depths:
        f.write('\\hline\n')
        f.write('&' + multicolumn(num_model_cols,'c|', f'depth={depth}') + '\\\\\n')
        f.write('\\hline\n')

        for dataset in allowed_datasets:
            if dataset == 'car_evaluation':
                dataset_name = 'car-evaluation'
            else:
                dataset_name = dataset

            row_data = []
            idx = 0
            for model in models:
                for _,_,df_f in model['Filtered dfs']:
                    row_entry = {'Solve Times': [],
                                 'Gaps': []}
                    row = df_f[(df_f['Dataset'] == dataset) & (df_f['depth'] == depth)]
                    if allowed_encodings is not None:
                        if 'Encoding Scheme' in df_f.columns:
                            row = row[row['Encoding Scheme'].isin(allowed_encodings)]
                        else:
                            print('Encoding Filter specified but dataset does not have an encoding scheme (probably categorical data)')

                    if len(row) == 0:
                        row_data.append('N/A')
                        idx += 1
                        continue

                    # Something has gone wrong if there are more rows than gurobi seeds
                    if len(row) > 1:
                        if 'Seed' not in row.columns:
                            assert False
                        else:
                            if len(row['Seed'].unique()) != len(row):
                                assert False

                    # Requires pd.options.mode.copy_on_write = True to be set
                    if 'Seed' not in row.columns:
                        row['Seed'] = 0

                    for seed in row['Seed'].unique():
                        row_per_seed = row[row['Seed'] == seed]

                        model_status = row_per_seed['Model Status'].item()

                        # TODO: Fix the super awkward method of accessing the model status
                        if model_status == 2:
                            # Solved to optimality
                            # model_solved = True
                            solve_time = row_per_seed['Solve Time'].item()
                            # if solve_time < best_time:
                            #     best_time = solve_time
                            #     best_model = idx
                            # row_entry['Solve Times'].append(f'{solve_time:.1f}')
                            row_entry['Solve Times'].append(solve_time)

                        elif model_status == 9:
                            # TimeLimit reached
                            solve_time = row_per_seed['Solve Time'].item()
                            if solve_time > 3650:
                                print(f'Found large solve time of {solve_time:0f} on dataset {dataset}')
                            gap = row_per_seed['Gap'].item()
                            row_entry['Gaps'].append(gap)
                            # row_entry['Gaps'].append(f'({gap*100:.2f})')
                            # if not model_solved:
                            #     if gap < best_gap:
                            #         best_gap = gap
                            #         best_model = idx
                        else:
                            row_data.append('N/A')

                        # idx += 1
                    num_solved = len(row_entry['Solve Times'])
                    num_unsolved = len(row_entry['Gaps'])
                    avg_solve_time = None if num_solved == 0 else sum(row_entry['Solve Times']) / num_solved
                    avg_gap = None if num_unsolved == 0 else sum(row_entry['Gaps']) / num_unsolved

                    row_data.append((avg_solve_time, avg_gap, num_solved, num_unsolved))

            model_solved = False
            best_model = -1
            best_time = 100000
            best_gap = 100

            row_data_str = []

            for idx, entry in enumerate(row_data):
                if entry == 'N/A':
                    row_data_str.append('N/A')
                    continue

                model_entry = '$'
                avg_solve_time, avg_gap, num_solved, num_unsolved = entry
                if avg_solve_time is not None:
                    model_solved = True

                    model_entry += f'{avg_solve_time:.1f}'
                    if avg_gap is not None:
                        model_entry += f'^{num_solved}'
                    if avg_solve_time < best_time:
                        best_model = idx
                        best_time = avg_solve_time

                if avg_gap is not None:
                    model_entry += f'({avg_gap*100:.2f})'
                    if not model_solved:
                        if avg_gap < best_gap:
                            best_gap = avg_gap
                            best_model = idx

                model_entry += '$'
                row_data_str.append(model_entry)


            if best_model != -1:
                row_data_str[best_model] = '$\\mathbf' + parenthesise(row_data_str[best_model][1:-1]) + '$'

            line = dataset_name + '&' + '&'.join(row_data_str) + '\\\\\n'

            f.write(line)

    f.write(postamble(tablename))

