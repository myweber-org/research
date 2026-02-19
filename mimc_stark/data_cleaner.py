
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Remove outliers
    if outlier_method == 'iqr':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
    elif outlier_method == 'zscore':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    # Normalize data
    if normalize_method == 'minmax':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
    elif normalize_method == 'zscore':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate statistical summary of the data.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        Dictionary with statistical summary
    """
    summary = {
        'shape': data.shape,
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_summary': {}
    }
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_summary'][col] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            '25%': data[col].quantile(0.25),
            '50%': data[col].quantile(0.50),
            '75%': data[col].quantile(0.75),
            'max': data[col].max(),
            'skewness': data[col].skew(),
            'kurtosis': data[col].kurtosis()
        }
    
    return summary