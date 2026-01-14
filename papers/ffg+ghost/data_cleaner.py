
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from specified column using IQR method.
    
    Args:
        df: pandas DataFrame
        column: column name to process
    
    Returns:
        Cleaned DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def normalize_column(df, column, method='minmax'):
    """
    Normalize specified column using selected method.
    
    Args:
        df: pandas DataFrame
        column: column name to normalize
        method: normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        df: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df_copy = df_copy.dropna(subset=numeric_cols)
    
    elif strategy == 'mean':
        for col in numeric_cols:
            if df_copy[col].isnull().any():
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
    
    elif strategy == 'median':
        for col in numeric_cols:
            if df_copy[col].isnull().any():
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
    
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    return df_copy

def clean_dataframe(df, numeric_columns=None, outlier_threshold=1.5, 
                   normalization_method='minmax', missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_threshold: IQR multiplier for outlier detection
        normalization_method: method for normalization
        missing_strategy: strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            cleaned_df = normalize_column(cleaned_df, column, method=normalization_method)
    
    return cleaned_df