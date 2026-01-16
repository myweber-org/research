
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns boolean mask for outliers.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, columns, threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    """
    clean_data = data.copy()
    for col in columns:
        outlier_mask = detect_outliers_iqr(clean_data, col, threshold)
        clean_data = clean_data[~outlier_mask]
    return clean_data.reset_index(drop=True)

def normalize_minmax(data, columns):
    """
    Apply Min-Max normalization to specified columns.
    Returns normalized copy of data.
    """
    normalized_data = data.copy()
    for col in columns:
        col_min = normalized_data[col].min()
        col_max = normalized_data[col].max()
        if col_max != col_min:
            normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
    return normalized_data

def standardize_zscore(data, columns):
    """
    Apply Z-score standardization to specified columns.
    Returns standardized copy of data.
    """
    standardized_data = data.copy()
    for col in columns:
        col_mean = standardized_data[col].mean()
        col_std = standardized_data[col].std()
        if col_std > 0:
            standardized_data[col] = (standardized_data[col] - col_mean) / col_std
    return standardized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    if columns is None:
        columns = data.columns
    
    processed_data = data.copy()
    
    if strategy == 'drop':
        return processed_data.dropna(subset=columns)
    
    for col in columns:
        if processed_data[col].isnull().any():
            if strategy == 'mean':
                fill_value = processed_data[col].mean()
            elif strategy == 'median':
                fill_value = processed_data[col].median()
            elif strategy == 'mode':
                fill_value = processed_data[col].mode()[0]
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            processed_data[col] = processed_data[col].fillna(fill_value)
    
    return processed_data

def validate_dataframe(data, required_columns=None, dtypes=None):
    """
    Validate dataframe structure and data types.
    Returns tuple of (is_valid, error_message)
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if dtypes:
        for col, expected_type in dtypes.items():
            if col in data.columns:
                actual_type = data[col].dtype
                if not np.issubdtype(actual_type, expected_type):
                    return False, f"Column {col} has type {actual_type}, expected {expected_type}"
    
    return True, "Data validation passed"