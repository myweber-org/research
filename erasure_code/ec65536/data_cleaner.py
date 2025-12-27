
import pandas as pd
import re

def clean_string_column(series, case='lower', strip=True, remove_special=True):
    """
    Standardize string values in a pandas Series.
    """
    if not pd.api.types.is_string_dtype(series):
        series = series.astype(str)
    
    if case == 'lower':
        series = series.str.lower()
    elif case == 'upper':
        series = series.str.upper()
    
    if strip:
        series = series.str.strip()
    
    if remove_special:
        series = series.apply(lambda x: re.sub(r'[^\w\s]', '', x))
    
    return series

def remove_duplicate_rows(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame with optional subset.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def standardize_numeric(series, fillna=0):
    """
    Convert series to numeric type and fill missing values.
    """
    series = pd.to_numeric(series, errors='coerce')
    return series.fillna(fillna)

def clean_dataframe(df, string_columns=None, numeric_columns=None):
    """
    Apply cleaning operations to multiple columns.
    """
    df_clean = df.copy()
    
    if string_columns:
        for col in string_columns:
            if col in df_clean.columns:
                df_clean[col] = clean_string_column(df_clean[col])
    
    if numeric_columns:
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = standardize_numeric(df_clean[col])
    
    return df_clean