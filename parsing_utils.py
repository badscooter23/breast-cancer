# Data parsing functions.
import pandas as pd
import numpy as np

def infer_schema(filename, nrows=1000):
    """
    Parse input file. Return downcasted results up to int8 and float32. 
    Also return decimal precision for float columns.

    filename: string  
        Input csv filename and location.
    
    nrows: int
        Number of rows to read to determine schema.

    """
    file_format = filename.split('.')[-1]

    if file_format == 'csv':
        df = pd.read_csv(filename, nrows=nrows)
        df_str = pd.read_csv(filename, nrows=nrows, dtype=str)
    elif file_format == 'parquet':
        df = pd.read_parquet(filename).iloc[:nrows,] # Needs to read entire file.
        df_str = df.astype(str)

    int_dtypes = df.select_dtypes('integer').columns.values
    float_dtypes = df.select_dtypes('float').columns.values

    # Downcast integers using pandas.to_numeric(). Limit is int8.
    for cc in int_dtypes:
        df[cc] = pd.to_numeric(df[cc], downcast='integer')
    
    # Downcast floats using pandas.to_numeric(). Limit is float32.
    # TODO: custom function to get float16. Or conversion to rounded float based on decimal precision.
    for cc in float_dtypes:
        df[cc] = pd.to_numeric(df[cc], downcast='float')

    numeric_schema = dict(df.select_dtypes('number').dtypes)

    # Determine decimal precision based on str in original csv.
    prec = {}
    for cc in float_dtypes:
        tmp_ = df_str[cc].str.split('.')

        # Get max number of decimal precision:
        prec[cc] = np.max([len(lst[1]) if len(lst)==2 else 0 for lst in tmp_])
    
    col_dtypes = {}
    col_dtypes['numeric'] = numeric_schema
    col_dtypes['precision'] = prec
    return(col_dtypes)


def apply_schema(df, col_dtypes, output_format):
    """
    Applies decimal schema to input dataframe.
    """
    if output_format == 'csv':
        df = df.astype(col_dtypes['numeric'])
        df = df.round(col_dtypes['precision'])
    else:
        df = df.astype(col_dtypes['numeric'])
    return(df)