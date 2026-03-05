
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
            cleaned_df = standardize_zscore(cleaned_df, col)
    return cleaned_df

def generate_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000)
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = generate_sample_data()
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    cleaned_data = clean_dataset(sample_df, numeric_cols)
    print(f"Original shape: {sample_df.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(cleaned_data.describe())import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df, column_type_map):
    """
    Convert columns to specified data types.
    
    Args:
        df: pandas DataFrame
        column_type_map: dict mapping column names to target types
    
    Returns:
        DataFrame with converted columns
    """
    df_copy = df.copy()
    for column, dtype in column_type_map.items():
        if column in df_copy.columns:
            df_copy[column] = df_copy[column].astype(dtype)
    return df_copy

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(fill_value)
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def normalize_column(df, column):
    """
    Normalize a column to range [0, 1].
    
    Args:
        df: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        return df
    
    df_copy = df.copy()
    col_min = df_copy[column].min()
    col_max = df_copy[column].max()
    
    if col_max != col_min:
        df_copy[column] = (df_copy[column] - col_min) / (col_max - col_min)
    
    return df_copy

def clean_dataframe(df, operations):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df: pandas DataFrame
        operations: list of dicts, each specifying an operation and parameters
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for operation in operations:
        op_type = operation.get('type')
        params = operation.get('params', {})
        
        if op_type == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, **params)
        elif op_type == 'convert_types':
            cleaned_df = convert_column_types(cleaned_df, **params)
        elif op_type == 'handle_missing':
            cleaned_df = handle_missing_values(cleaned_df, **params)
        elif op_type == 'normalize':
            cleaned_df = normalize_column(cleaned_df, **params)
    
    return cleaned_df