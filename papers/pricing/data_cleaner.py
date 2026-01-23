import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def detect_skewed_columns(data, threshold=0.5):
    """
    Detect columns with skewed distributions.
    
    Args:
        data: pandas DataFrame
        threshold: absolute skewness threshold (default 0.5)
    
    Returns:
        List of column names with skewness above threshold
    """
    skewed_cols = []
    
    for col in data.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(data[col].dropna())
        if abs(skewness) > threshold:
            skewed_cols.append(col)
    
    return skewed_cols

def log_transform(data, column):
    """
    Apply log transformation to reduce skewness.
    
    Args:
        data: pandas DataFrame
        column: column name to transform
    
    Returns:
        Series with log-transformed values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    if data[column].min() <= 0:
        # Add constant to handle zero or negative values
        constant = abs(data[column].min()) + 1
        transformed = np.log(data[column] + constant)
    else:
        transformed = np.log(data[column])
    
    return transformed

def clean_dataset(data, numeric_columns=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove outliers from each numeric column
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    # Standardize numeric columns
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data[f'{col}_standardized'] = standardize_zscore(cleaned_data, col)
    
    return cleaned_data

def validate_data(data, required_columns=None):
    """
    Validate data quality.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'duplicate_rows': data.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        validation_results['missing_required_columns'] = missing_columns
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    validation_results['numeric_columns'] = len(numeric_cols)
    
    return validation_results