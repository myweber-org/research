import numpy as np
import pandas as pd

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Parameters:
    df: pandas DataFrame
    columns: list of column names to process
    factor: IQR multiplier (default 1.5)
    
    Returns:
    DataFrame with outliers removed
    """
    df_clean = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns):
    """
    Normalize columns using min-max scaling.
    
    Parameters:
    df: pandas DataFrame
    columns: list of column names to normalize
    
    Returns:
    DataFrame with normalized columns
    """
    df_norm = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
            
        min_val = df[col].min()
        max_val = df[col].max()
        
        if max_val != min_val:
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0
    
    return df_norm

def standardize_zscore(df, columns):
    """
    Standardize columns using z-score normalization.
    
    Parameters:
    df: pandas DataFrame
    columns: list of column names to standardize
    
    Returns:
    DataFrame with standardized columns
    """
    df_std = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
            
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        if std_val > 0:
            df_std[col] = (df[col] - mean_val) / std_val
        else:
            df_std[col] = 0
    
    return df_std

def clean_dataset(df, numeric_columns, outlier_factor=1.5, method='standardize'):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    df: pandas DataFrame
    numeric_columns: list of numeric column names
    outlier_factor: IQR factor for outlier removal
    method: normalization method ('standardize', 'normalize', or None)
    
    Returns:
    Cleaned DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if not numeric_columns:
        return df.copy()
    
    df_clean = remove_outliers_iqr(df, numeric_columns, outlier_factor)
    
    if method == 'standardize':
        df_clean = standardize_zscore(df_clean, numeric_columns)
    elif method == 'normalize':
        df_clean = normalize_minmax(df_clean, numeric_columns)
    
    return df_clean

def validate_cleaning(df_original, df_cleaned, numeric_columns):
    """
    Validate cleaning results by comparing statistics.
    
    Parameters:
    df_original: original DataFrame
    df_cleaned: cleaned DataFrame
    numeric_columns: list of numeric column names
    
    Returns:
    Dictionary with validation metrics
    """
    validation = {}
    
    for col in numeric_columns:
        if col not in df_original.columns:
            continue
            
        original_stats = {
            'count': len(df_original[col]),
            'mean': df_original[col].mean(),
            'std': df_original[col].std(),
            'min': df_original[col].min(),
            'max': df_original[col].max()
        }
        
        cleaned_stats = {
            'count': len(df_cleaned[col]),
            'mean': df_cleaned[col].mean(),
            'std': df_cleaned[col].std(),
            'min': df_cleaned[col].min(),
            'max': df_cleaned[col].max()
        }
        
        validation[col] = {
            'original': original_stats,
            'cleaned': cleaned_stats,
            'rows_removed': original_stats['count'] - cleaned_stats['count'],
            'rows_removed_pct': ((original_stats['count'] - cleaned_stats['count']) / original_stats['count']) * 100
        }
    
    return validation