import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process, None for all numeric columns
    factor (float): Multiplier for IQR
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize, None for all numeric columns
    feature_range (tuple): Desired range of transformed data
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max - col_min != 0:
                df_normalized[col] = min_val + (df[col] - col_min) * (max_val - min_val) / (col_max - col_min)
            else:
                df_normalized[col] = min_val
    
    return df_normalized

def zscore_normalize(df, columns=None, threshold=3):
    """
    Normalize data using Z-score and optionally remove extreme outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process
    threshold (float): Z-score threshold for outlier removal
    
    Returns:
    pd.DataFrame: Normalized dataframe with extreme outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            mask = z_scores < threshold
            
            valid_indices = df[col].dropna().index[mask]
            df_normalized.loc[valid_indices, col] = stats.zscore(df.loc[valid_indices, col])
            
            df_normalized.loc[~df.index.isin(valid_indices), col] = np.nan
    
    return df_normalized

def clean_dataset(df, outlier_method='iqr', normalize_method='minmax', **kwargs):
    """
    Main function to clean and normalize dataset.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    outlier_method (str): 'iqr', 'zscore', or None
    normalize_method (str): 'minmax', 'zscore', or None
    **kwargs: Additional arguments for specific methods
    
    Returns:
    pd.DataFrame: Cleaned and normalized dataframe
    """
    df_clean = df.copy()
    
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df_clean, **kwargs)
    elif outlier_method == 'zscore':
        threshold = kwargs.get('threshold', 3)
        df_clean = zscore_normalize(df_clean, threshold=threshold)
    
    if normalize_method == 'minmax':
        df_clean = normalize_minmax(df_clean, **kwargs)
    elif normalize_method == 'zscore':
        df_clean = zscore_normalize(df_clean, **kwargs)
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=10):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"Dataframe has less than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return False, "No numeric columns found in dataframe"
    
    return True, "Dataframe validation passed"