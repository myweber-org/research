
import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to check for missing values.
    
    Returns:
        pd.DataFrame: DataFrame with rows containing missing values removed
    """
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def fill_missing_with_mean(df, columns=None):
    """
    Fill missing values with column mean.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to fill.
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled
    """
    df_filled = df.copy()
    
    if columns:
        for col in columns:
            if col in df.columns:
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    else:
        for col in df_filled.columns:
            if df_filled[col].dtype in [np.float64, np.int64]:
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    
    return df_filled

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to check for outliers
        factor (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns):
    """
    Standardize specified columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    df_std = df.copy()
    
    for col in columns:
        if col in df_std.columns:
            mean_val = df_std[col].mean()
            std_val = df_std[col].std()
            
            if std_val > 0:
                df_std[col] = (df_std[col] - mean_val) / std_val
    
    return df_std

def clean_dataset(df, missing_strategy='remove', outlier_columns=None, standardize_columns_list=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): 'remove' or 'mean' for handling missing values
        outlier_columns (list, optional): Columns to remove outliers from
        standardize_columns_list (list, optional): Columns to standardize
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df)
    elif missing_strategy == 'mean':
        cleaned_df = fill_missing_with_mean(cleaned_df)
    
    if outlier_columns:
        cleaned_df = remove_outliers_iqr(cleaned_df, outlier_columns)
    
    if standardize_columns_list:
        cleaned_df = standardize_columns(cleaned_df, standardize_columns_list)
    
    return cleaned_df