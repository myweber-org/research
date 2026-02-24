
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range (IQR) method.
    
    Args:
        data: pandas DataFrame containing the data
        column: string name of the column to process
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    outliers_removed = len(data) - len(filtered_data)
    print(f"Removed {outliers_removed} outliers from column '{column}'")
    
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column.
    
    Args:
        data: pandas DataFrame
        column: string name of the column
    
    Returns:
        Dictionary containing summary statistics
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'count': data[column].count()
    }
    
    return stats

def normalize_column(data, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        data: pandas DataFrame
        column: string name of the column to normalize
        method: string normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column added as new column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    if method == 'minmax':
        min_val = data[column].min()
        max_val = data[column].max()
        normalized = (data[column] - min_val) / (max_val - min_val)
        new_col_name = f"{column}_normalized_minmax"
    
    elif method == 'zscore':
        mean_val = data[column].mean()
        std_val = data[column].std()
        normalized = (data[column] - mean_val) / std_val
        new_col_name = f"{column}_normalized_zscore"
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    data[new_col_name] = normalized
    return data