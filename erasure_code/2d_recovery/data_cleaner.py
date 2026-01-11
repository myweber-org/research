import pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize
        method: 'minmax' or 'zscore' normalization
    
    Returns:
        DataFrame with normalized columns
    """
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        if method == 'minmax':
            min_val = df_copy[col].min()
            max_val = df_copy[col].max()
            if max_val > min_val:
                df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = df_copy[col].mean()
            std_val = df_copy[col].std()
            if std_val > 0:
                df_copy[col] = (df_copy[col] - mean_val) / std_val
    
    return df_copy

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
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        if method == 'iqr':
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)
            df_copy = df_copy[mask]
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df_copy[col].dropna()))
            mask = z_scores < threshold
            df_copy = df_copy[mask]
    
    return df_copy.reset_index(drop=True)

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
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        if strategy == 'mean':
            df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        elif strategy == 'median':
            df_copy[col].fillna(df_copy[col].median(), inplace=True)
        elif strategy == 'mode':
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        elif strategy == 'drop':
            df_copy = df_copy.dropna(subset=[col])
    
    return df_copy

def validate_data(df, column_types):
    """
    Validate DataFrame columns against expected data types.
    
    Args:
        df: pandas DataFrame
        column_types: dict mapping column names to expected types
    
    Returns:
        dict with validation results
    """
    validation_results = {}
    
    for col, expected_type in column_types.items():
        if col not in df.columns:
            validation_results[col] = {'status': 'missing', 'message': 'Column not found'}
            continue
        
        actual_type = str(df[col].dtype)
        
        if expected_type == 'numeric':
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            validation_results[col] = {
                'status': 'valid' if is_numeric else 'invalid',
                'expected': expected_type,
                'actual': actual_type
            }
        elif expected_type == 'datetime':
            try:
                pd.to_datetime(df[col])
                validation_results[col] = {
                    'status': 'valid',
                    'expected': expected_type,
                    'actual': actual_type
                }
            except:
                validation_results[col] = {
                    'status': 'invalid',
                    'expected': expected_type,
                    'actual': actual_type
                }
        else:
            validation_results[col] = {
                'status': 'valid' if actual_type == expected_type else 'invalid',
                'expected': expected_type,
                'actual': actual_type
            }
    
    return validation_resultsimport pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """Fill missing values using specified strategy."""
    if columns is None:
        columns = df.columns
    
    df_filled = df.copy()
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            df_filled[col].fillna(fill_value, inplace=True)
        else:
            df_filled[col].fillna('Unknown', inplace=True)
    return df_filled

def normalize_column(df, column):
    """Normalize numeric column to range [0,1]."""
    if df[column].dtype in [np.float64, np.int64]:
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataframe(df, operations=None):
    """Apply multiple cleaning operations to DataFrame."""
    if operations is None:
        operations = ['remove_duplicates', 'fill_missing']
    
    cleaned_df = df.copy()
    
    if 'remove_duplicates' in operations:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if 'fill_missing' in operations:
        cleaned_df = fill_missing_values(cleaned_df)
    
    return cleaned_df