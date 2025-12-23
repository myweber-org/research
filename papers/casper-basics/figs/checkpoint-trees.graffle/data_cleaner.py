
import pandas as pd
import numpy as np

def remove_missing_rows(df, threshold=0.5):
    """
    Remove rows with missing values exceeding threshold percentage.
    
    Args:
        df: pandas DataFrame
        threshold: float between 0 and 1, default 0.5
    
    Returns:
        Cleaned DataFrame
    """
    missing_per_row = df.isnull().mean(axis=1)
    return df[missing_per_row <= threshold]

def fill_missing_with_median(df, columns=None):
    """
    Fill missing values with column median.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
    
    Returns:
        DataFrame with filled values
    """
    df_filled = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            median_val = df[col].median()
            df_filled[col].fillna(median_val, inplace=True)
    
    return df_filled

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using IQR method.
    
    Args:
        df: pandas DataFrame
        column: column name to check for outliers
        multiplier: IQR multiplier, default 1.5
    
    Returns:
        Boolean Series indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def cap_outliers(df, column, method='iqr', multiplier=1.5):
    """
    Cap outliers to specified bounds.
    
    Args:
        df: pandas DataFrame
        column: column name to process
        method: 'iqr' or 'percentile'
        multiplier: IQR multiplier if method='iqr'
    
    Returns:
        DataFrame with capped values
    """
    df_capped = df.copy()
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
    elif method == 'percentile':
        lower_bound = df[column].quantile(0.01)
        upper_bound = df[column].quantile(0.99)
    
    else:
        raise ValueError("Method must be 'iqr' or 'percentile'")
    
    df_capped[column] = df_capped[column].clip(lower=lower_bound, upper=upper_bound)
    return df_capped

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df: pandas DataFrame
        column: column name to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized column
    """
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_normalized

def clean_dataset(df, missing_threshold=0.3, outlier_columns=None):
    """
    Comprehensive dataset cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        missing_threshold: threshold for removing rows with missing values
        outlier_columns: list of columns to check for outliers
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    cleaned_df = remove_missing_rows(cleaned_df, threshold=missing_threshold)
    cleaned_df = fill_missing_with_median(cleaned_df)
    
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = cap_outliers(cleaned_df, col, method='iqr')
    
    return cleaned_df