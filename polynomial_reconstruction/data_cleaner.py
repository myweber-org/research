import pandas as pd
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

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in DataFrame columns.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: list of columns to fill, None for all columns
    
    Returns:
        DataFrame with missing values filled
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
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize numerical columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to normalize, None for all numerical columns
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def detect_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Detect outliers in numerical columns.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to check, None for all numerical columns
        method: 'iqr' for interquartile range method
        threshold: multiplier for IQR method
    
    Returns:
        DataFrame with boolean columns indicating outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outlier_df = pd.DataFrame(index=df.index)
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_col = f"{col}_outlier"
            outlier_df[outlier_col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    return outlier_df

def clean_dataframe(df, remove_dups=True, fill_na=True, normalize=False):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        remove_dups: whether to remove duplicates
        fill_na: whether to fill missing values
        normalize: whether to normalize numerical columns
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df, strategy='mean')
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df, method='minmax')
    
    return cleaned_df