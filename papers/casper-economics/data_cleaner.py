import numpy as np

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers from a dataset using the Interquartile Range method.
    
    Args:
        data: numpy array or list of numerical values
        column: index or key to access the data column
        multiplier: IQR multiplier for outlier detection (default: 1.5)
    
    Returns:
        Cleaned data without outliers
    """
    if isinstance(data, list):
        data = np.array(data)
    
    q1 = np.percentile(data[:, column], 25)
    q3 = np.percentile(data[:, column], 75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    mask = (data[:, column] >= lower_bound) & (data[:, column] <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a data column.
    
    Args:
        data: numpy array
        column: index of the column to analyze
    
    Returns:
        Dictionary containing mean, median, std, min, max
    """
    column_data = data[:, column]
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data),
        'count': len(column_data)
    }
    
    return stats

def normalize_data(data, column, method='minmax'):
    """
    Normalize data in a column using specified method.
    
    Args:
        data: numpy array
        column: index of the column to normalize
        method: normalization method ('minmax' or 'zscore')
    
    Returns:
        Data with normalized column
    """
    column_data = data[:, column].astype(float)
    
    if method == 'minmax':
        min_val = np.min(column_data)
        max_val = np.max(column_data)
        if max_val - min_val != 0:
            normalized = (column_data - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(column_data)
    
    elif method == 'zscore':
        mean = np.mean(column_data)
        std = np.std(column_data)
        if std != 0:
            normalized = (column_data - mean) / std
        else:
            normalized = np.zeros_like(column_data)
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    data[:, column] = normalized
    return data

def validate_data(data, column, value_range=None):
    """
    Validate data in a column against specified criteria.
    
    Args:
        data: numpy array
        column: index of the column to validate
        value_range: tuple of (min, max) allowed values
    
    Returns:
        Boolean mask indicating valid rows
    """
    column_data = data[:, column]
    
    mask = np.ones(len(data), dtype=bool)
    
    if value_range is not None:
        min_val, max_val = value_range
        mask = mask & (column_data >= min_val) & (column_data <= max_val)
    
    mask = mask & ~np.isnan(column_data)
    mask = mask & ~np.isinf(column_data)
    
    return mask