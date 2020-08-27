#!/usr/bin/env python3

"""
Description
   generate an arbitrary nubmer of data files (part files) based on a small breast cancer dataset.

   This python script uses an over sampling and re-balancng technique (based on loosely on this


Purpose

Created on Sun Mar 3 12:08:00 2019

@author: badscooter23 (scott.j.mcclellan@gmail.com)
"""

# import various libraries - these are all used in the Jupyter notebook
import pandas as pd
import os
import numpy as np
import sweetviz as sv
import imblearn

# additional libraries we use for the .py code (not currently used in Jupyter notebook)
import argparse

# defaults for cmd line parms...
DEFAULT_PART_FILES = 10
DEFAULT_N_FACTOR = 100
DEFAULT_VERBOSE = False
verbose_global = DEFAULT_VERBOSE


def setup_environment_variables():
    # set cwd variables
    cwd = os.getcwd()
    print('cwd: {}'.format(cwd))

    data_dir = os.path.join(cwd, 'data')
    if os.path.isdir(data_dir):
        print('data_dir: {}'.format(data_dir))
    else:
        print('oops! directory named "data" not found under "{}"'.format(cwd))
        data_dir = os.path.join(cwd, 'data')

    eda_dir = os.path.join(cwd, 'EDA')
    if os.path.isdir(eda_dir):
        print('eda_dir: {}'.format(eda_dir))
    else:
        print('directory named "EDA" not found under "{}"'.format(cwd))
        print('creating "EDA" dir... "{}"'.format(eda_dir))
        os.makedirs(eda_dir)

    part_dir = os.path.join(cwd, 'part-files')
    if os.path.isdir(part_dir):
        print('part_dir: {}'.format(part_dir))
    else:
        print('directory named "part-files" not found under "{}"'.format(cwd))
        print('creating "part-files" dir... "{}"'.format(part_dir))
        os.makedirs(part_dir)

    return cwd, data_dir, eda_dir, part_dir


def now():
    from datetime import datetime
    return datetime.now().strftime("%d%m%Y-%H:%M:%S")


def name_df(df, name, desc=""):
    from datetime import date
    if desc == "":
        df.name = "".join((name, "-", now()))
    else:
        df.name = "".join((name, "-", now(), "-(", desc, ")"))
    return name


def create_initial_cancer_dataset():
    # open the  cancer data file
    cancer_dataset_name = 'cancer_data'
    cancer_df = pd.read_csv(os.path.join(cwd, data_dir, cancer_dataset_name + ".csv"))

    # convert 'diagnosis' column to a categorical
    cancer_df['diagnosis'] = pd.Categorical(cancer_df['diagnosis'], cancer_categories, ordered=True).codes
    cancer_df = cancer_df.drop(columns=['id'])

    name_df(cancer_df, 'cancer_df', 'Original Cancer Data')

    return cancer_df, cancer_dataset_name


def create_imbalanced_dataset(df, over_balance_on, N=100, verbose=False):

    # replicate the starting data frame (df) N times into df2
    if verbose:
        print('replicating base dataframe {} times'.format(N))
    df2 = pd.concat([df for ii in range(N)])

    if verbose:
        print('original dataframe: {} rows, new/temp dataframe: {} rows\n'.format(len(df), len(df2)))

    # assuming (for now) that we are balancing relative to a 'diagnosis' (that is binary classification: 0 or 1)
    # validate the the 'over_balance_on' parm ..
    if over_balance_on == 0:
        minority = 1
    elif over_balance_on == 1:
        minority = 0
    else:
        print("ERROR: over_balance_on has to be 0 or 1 (binary classificaion only)!")
        return
    # print("valid 'over_balance_on' parameter specified... ")

    majority = over_balance_on
    # minority_st = cancer_categories[minority]
    over_balance_on_st = cancer_categories[over_balance_on]

    print('creating a new dataframe imbalanced on ''diagnosis=="{}"'' ({})'.format(over_balance_on_st, over_balance_on))

    # create a new dataframe 'majority_df' by selecting rows where 'diagnosis==majority' from the
    # temporary dataframe (which was replicated Nx from the base_df)
    majority_df = df2.query('diagnosis=={}'.format(majority))

    # majority_rows = len(majority_df)
    # print('... {} rows - containing ''diagnosis=="{}"'' only'.format(majority_rows, over_balance_on_st))
    # print('... added to {} total rose - containing a mix of ''diagnosis''\n'.format(len(df)))

    imbalanced_df = df.append(majority_df)

    return imbalanced_df


def print_balance_stats(df):

    b_rows = len(df.query('diagnosis=={}'.format(B)))
    m_rows = len(df.query('diagnosis=={}'.format(M)))
    t_rows = len(df)
    if m_rows > b_rows:
        print("dataframe is over balanced toward '{}' ({:.2F}%)".format(cancer_categories[M], (m_rows / t_rows) * 100))
    elif b_rows > m_rows:
        print("dataframe is over balanced toward '{}' ({:.2F}%)".format(cancer_categories[M], (b_rows / t_rows) * 100))
    else:
        print("the dataframe is balanced!")
    print("B: {}, M: {}, total: {}  ({})".format(b_rows, m_rows, t_rows, (m_rows + b_rows) == t_rows))

    return b_rows, m_rows, t_rows


def balance_dataset(df, verbose=False):

    # pass 'balance_dataset' a dataframe that should ideally be imbalanced and 'balance_dataset'
    # will apply Synthetic Minority Over-sampling Technique (aka: SMOTE) to reblance the data
    #
    # the re-balancing technique involves breaking the dataframe into
    #     y    a 'target_vector' which is essentially the 'diagnosis' column from 'df'
    #     X    the features matrix which is essentially all the remaining columns in the matrix

    if verbose:
        print("initial balance statistics (before re-balancing)")
        print_balance_state(df)

    # separate the feature matrix (X) from the 'target vector' (y)
    # WARNING: code below assumes that the 'diagnosis', it the first column () in the dataframe
    # should re-write it to work regardless of column order...
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    # apply Synthetic Minority Over-sampling Technique (aka: SMOTE) to reblance the data
    # (creating a 50/50 ratio of malignant and benign cases)

    # note: SMOTE will return "re-sampled" versions of X and y that have additional entries created
    # to achieve balance
    oversample = imblearn.over_sampling.SMOTE()
    X_resamp, y_resamp = oversample.fit_resample(X, y)

    # reassemble the dataframe into 'rebalanced_df' (which will be returned from the function)

    # build a list of column names
    column_names = list(cancer_df.columns)
    if verbose:
        print(column_names)

    # reassemble the dataframe from X_reasmp and y_resamp
    rebalanced_df = pd.DataFrame(np.insert(X_resamp, 0, y_resamp, axis=1), columns=column_names)

    if verbose:
        print("There sould be an equal number of 'benign' and 'malignant' cases after rebalancing... ")
        print("benign:", len(rebalanced_df.query("diagnosis=={}".format(B))))
        print("malignant:", len(rebalanced_df.query("diagnosis=={}".format(M))))

    return rebalanced_df


def gen_new_data(N, P, dataset_name, verbose=verbose_global):

    for i in range(P):
        malignant_imbalanced = create_imbalanced_dataset(cancer_df, M, N)
        print('malignant_imbalanced: should have M >> B')
        _, _, _ = print_balance_stats(malignant_imbalanced)

        print('\nrebalanced_df: should have M == B')
        rebalanced_df = balance_dataset(malignant_imbalanced)
        _, _, _ = print_balance_stats(rebalanced_df)
        new_df = rebalanced_df.query('diagnosis=={}'.format(B))

        benign_imbalanced = create_imbalanced_dataset(cancer_df, B, N)
        print('\nmalignant_imbalanced: should have B >> M')
        _, _, _ = print_balance_stats(benign_imbalanced)

        if verbose:
            print('\nrebalanced_df: should have B == M')
        rebalanced_df = balance_dataset(malignant_imbalanced)
        _, _, _ = print_balance_stats(rebalanced_df)
        new_df = new_df.append(rebalanced_df.query('diagnosis=={}'.format(M)))

        pf_name = os.path.join(part_dir, '{}-{}.csv'.format(dataset_name, str(i).zfill(5)))
        print('\n*** new partfile: {}\n'.format(pf_name))
        new_df.to_csv(pf_name, index=False)


def parse_args():

    arg_parser, args = setup_arg_parser()

    # setup my_args... dictionary...
    my_args = {}

    my_args['verbose'] = args.verbose
    my_args['skip_flag'] = args.skip_flag

    my_args['part_files'] = args.part_files
    if not my_args['part_files'].isnumeric():
        say.error("--part_files should be a numeric value! Using default value ({}).".format(DEFAULT_PART_FILES))
        my_args['part_size'] = str(DEFAULT_PART_FILES)
    my_args['part_files_value'] = int(my_args['part_files'])

    my_args['num_copies'] = args.num_copies
    if not my_args['num_copies'].isnumeric():
        say.error("--num_copies numeric value! Using default value ({}).".format(DEFAULT_N_FACTOR))
        my_args['num_copies'] = str(DEFAULT_N_FACTOR)
    my_args['num_copies_value'] = int(my_args['num_copies'])

    # for now set the skip_flag to True if skip_flag is true (so it will dump mayArgs...)
    # useful for debugging cmdline argument parsing
    skip_flag = my_args['skip_flag']
    if skip_flag:
        print("myArg[] values for 'non-flag' parameters...")
        print("---------------------------------------------------------")
        print("my_args['part_files']: '{}'".format(my_args['part_files']))
        print("my_args['part_files_value']: {}".format(my_args['part_files_value']))
        print("my_args['num_copies']: '{}'".format(my_args['num_copies']))
        print("my_args['num_copies_value']: '{}'".format(my_args['num_copies_value']))

        print()
        print("myArg[] values for 'flag' parameters...")
        print("---------------------------------------------------------")
        print("my_args['verbose']: ", my_args['verbose'], sep="")
        print("my_args['skip_flag']: ", my_args['skip_flag'], sep="")
        exit(0)

    return my_args


def setup_arg_parser():
    arg_parser = argparse.ArgumentParser(description="Utility program to generate large amounts ...",
                                         prog='gendata')

    arg_parser.add_argument('--num_copies', '-N',
                            help="Number of copies of the original dataset that will be made to 'seed' the data generation. (Default: {})".format(DEFAULT_N_FACTOR),
                            required=False, default=str(DEFAULT_N_FACTOR))

    arg_parser.add_argument('--part_files', '-P',
                            help="Number of part files that will be generated from original data. (Default: {})".format(DEFAULT_PART_FILES),
                            required=False, default=str(DEFAULT_PART_FILES))

    # output mode
    arg_parser.add_argument('--verbose', '-v',
                            help="Verbose flag: 'lb' will print out more verbose messages.",
                            required=False, action="store_true", default=DEFAULT_VERBOSE)

    # skip?
    arg_parser.add_argument('--skip_flag', '-Z',
                            help="Skip all processing - useful for debugging argument parsing. (default=False)",
                            required=False, action="store_true", default=False)

    args = arg_parser.parse_args()

    return arg_parser, args


if __name__ == "__main__":

    # parse cmd line arguments...
    my_args = dict({})
    my_args = parse_args()

    # set global flag for output control
    verbose_global = my_args['verbose']

    P = my_args['part_files_value']
    N = my_args['num_copies_value']

    # initialize global environment variables ...
    cwd, data_dir, eda_dir, part_dir = setup_environment_variables()

    # setup 'cancer_categories' to be used to convert 'B' and 'M' into categorical (numeric) values
    cancer_categories = ['B', 'M']
    # remember the indices for B and M (for use in other functions, etc)
    B = cancer_categories.index('B')
    M = cancer_categories.index('M')

    # initialize cancer_df from the raw data file
    cancer_df, cancer_dataset_name = create_initial_cancer_dataset()
    print('cancer_df.name: "{}"'.format(cancer_df.name))

    gen_new_data(N, P, cancer_dataset_name)
