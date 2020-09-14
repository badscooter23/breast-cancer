# Data parsing functions.
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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

    # Determine decimal format based on str in original csv.
    decimal_fmt = {}
    for cc in float_dtypes:
        # Get max number of decimal for scale (number of digits to the right of the decimal point):
        scale = np.max([len(lst[1]) if len(lst)==2 else 0 for lst in df_str[cc].str.split('.')])

        # Count number of digits to left of decimal: 
        left_dig = np.max([len(lst[0]) if len(lst)==2 else 0 for lst in df_str[cc].str.split('.')])

        # Get precision (total number of digits) based on combined field length:
        precision = scale + left_dig + 1

        decimal_fmt[cc] = (precision, scale)
        # TODO: allow integers (scale=0)?
    
    col_dtypes = {}
    col_dtypes['numeric'] = numeric_schema
    col_dtypes['decimal_fmt'] = decimal_fmt
    return(col_dtypes)


def apply_schema(df, col_dtypes, output_format):
    """
    Applies decimal schema to input dataframe.
    """
    if output_format in ['csv', 'csv.gz']:
        df = df.astype(col_dtypes['numeric'])
        df = df.round(col_dtypes['scale'])

    elif output_format == 'parquet':
        # Apply downcasted schema:
        df = df.astype(col_dtypes['numeric'])
        
        # Convert to pyarrow format to access decimal128 dtype. Pandas does not handle decimals well.
        df_pa = pa.Table.from_pandas(df)

        fields = []
        for cc in df_pa.column_names:
            if cc in col_dtypes['decimal_fmt'].keys():
                dec_fmt = pa.decimal128(*col_dtypes['decimal_fmt'][cc])
                fields.append(pa.field(cc, dec_fmt))
            else:
                fields.append(df_pa.schema.field_by_name(cc))

        updated_schema = pa.schema(fields)

        # Apply schema to arrow Table. Need to use pyarrow.parquet to write file out.
        df = df_pa.cast(updated_schema)
    else:
        raise ValueError('Case not implemented. Choose output_format of csv, csv.gz, or parquet.')
    return(df)