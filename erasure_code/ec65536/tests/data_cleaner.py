import pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns, method='zscore'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize
        method: 'zscore' or 'minmax' normalization
    
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'zscore':
            df_normalized[col] = stats.zscore(df[col])
        elif method == 'minmax':
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df_normalized[col] = 0
    
    return df_normalized

def remove_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process
        method: 'iqr' or 'zscore' outlier detection
        threshold: multiplier for IQR or z-score cutoff
    
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col]))
            mask = z_scores < threshold
        
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def clean_missing_values(df, columns, strategy='mean'):
    """
    Handle missing values in specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process
        strategy: 'mean', 'median', 'mode', or 'drop'
    
    Returns:
        DataFrame with handled missing values
    """
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean':
            fill_value = df[col].mean()
        elif strategy == 'median':
            fill_value = df[col].median()
        elif strategy == 'mode':
            fill_value = df[col].mode()[0] if not df[col].mode().empty else np.nan
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
            continue
        
        df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

def validate_data(df, check_types=True, check_ranges=None):
    """
    Validate DataFrame for common data quality issues.
    
    Args:
        df: pandas DataFrame
        check_types: verify column data types
        check_ranges: dict of {column: (min, max)} for range validation
    
    Returns:
        dict with validation results
    """
    validation_results = {
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'data_types': {}
    }
    
    if check_types:
        validation_results['data_types'] = df.dtypes.astype(str).to_dict()
    
    if check_ranges:
        validation_results['range_violations'] = {}
        for col, (min_val, max_val) in check_ranges.items():
            if col in df.columns:
                violations = ((df[col] < min_val) | (df[col] > max_val)).sum()
                validation_results['range_violations'][col] = violations
    
    return validation_results