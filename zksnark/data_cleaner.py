
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (bool): Whether to fill missing values. Default is False.
        fill_value: Value to use for filling missing values. Default is 0.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
        method (str): Method for outlier detection ('iqr' or 'zscore').
        threshold (float): Threshold for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'iqr':
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)
    
    elif method == 'zscore':
        mean = df_copy[column].mean()
        std = df_copy[column].std()
        z_scores = (df_copy[column] - mean) / std
        mask = abs(z_scores) <= threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df_copy[mask]

def normalize_column(df, column, method='minmax'):
    """
    Normalize values in a specific column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to normalize.
        method (str): Normalization method ('minmax' or 'zscore').
    
    Returns:
        pd.DataFrame: DataFrame with normalized column.
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
        mean = df_copy[column].mean()
        std = df_copy[column].std()
        if std != 0:
            df_copy[column] = (df_copy[column] - mean) / std
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy