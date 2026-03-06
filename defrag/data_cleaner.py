
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in specified columns using a given strategy.
    """
    df_filled = df.copy()
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            df_filled[col].fillna(fill_value, inplace=True)
        else:
            df_filled[col].fillna('Unknown', inplace=True)
    return df_filled

def validate_numeric_range(df, column, min_val=None, max_val=None):
    """
    Validate that values in a numeric column fall within a specified range.
    """
    if min_val is not None:
        invalid_min = df[column] < min_val
        if invalid_min.any():
            print(f"Warning: {invalid_min.sum()} values below minimum {min_val} in column {column}")
    
    if max_val is not None:
        invalid_max = df[column] > max_val
        if invalid_max.any():
            print(f"Warning: {invalid_max.sum()} values above maximum {max_val} in column {column}")
    
    return df[(df[column] >= (min_val if min_val else -np.inf)) & 
              (df[column] <= (max_val if max_val else np.inf))]

def clean_column_names(df):
    """
    Standardize column names to lowercase with underscores.
    """
    df_clean = df.copy()
    df_clean.columns = [col.lower().replace(' ', '_') for col in df_clean.columns]
    return df_clean

def get_summary_statistics(df):
    """
    Generate summary statistics for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return df[numeric_cols].describe()
    else:
        return pd.DataFrame()
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result