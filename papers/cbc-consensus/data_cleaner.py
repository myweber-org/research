import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from specified columns or entire DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of columns to check. Defaults to None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def fill_missing_with_mean(df, columns):
    """
    Fill missing values in specified columns with column mean.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to fill
    
    Returns:
        pd.DataFrame: DataFrame with filled values
    """
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    return df_filled

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame without outliers
    """
    if column not in df.columns:
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def standardize_column(df, column):
    """
    Standardize a column using z-score normalization.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized column
    """
    if column not in df.columns:
        return df
    
    df_standardized = df.copy()
    mean_val = df_standardized[column].mean()
    std_val = df_standardized[column].std()
    
    if std_val > 0:
        df_standardized[column] = (df_standardized[column] - mean_val) / std_val
    
    return df_standardized

def clean_dataframe(df, missing_strategy='remove', outlier_columns=None):
    """
    Comprehensive data cleaning function.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values
        outlier_columns (list): Columns to check for outliers
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df)
    elif missing_strategy == 'mean':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        cleaned_df = fill_missing_with_mean(cleaned_df, numeric_cols)
    
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    return cleaned_df