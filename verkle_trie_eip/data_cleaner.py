
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_data(data, column):
    """
    Standardize data using Z-score normalization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_missing_values(data, strategy='mean'):
    """
    Handle missing values with different strategies
    """
    cleaned_data = data.copy()
    
    for column in cleaned_data.select_dtypes(include=[np.number]).columns:
        if cleaned_data[column].isna().any():
            if strategy == 'mean':
                fill_value = cleaned_data[column].mean()
            elif strategy == 'median':
                fill_value = cleaned_data[column].median()
            elif strategy == 'mode':
                fill_value = cleaned_data[column].mode()[0]
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            cleaned_data[column] = cleaned_data[column].fillna(fill_value)
    
    return cleaned_data

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def create_data_summary(df):
    """
    Create comprehensive data summary
    """
    summary = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isna().sum().to_dict(),
        'numeric_stats': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {},
        'categorical_stats': {col: df[col].value_counts().to_dict() 
                            for col in df.select_dtypes(include=['object']).columns}
    }
    return summary
import numpy as np
import pandas as pd

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

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data
    
    data_copy = data.copy()
    data_copy[column] = (data_copy[column] - min_val) / (max_val - min_val)
    return data_copy

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data
    
    data_copy = data.copy()
    data_copy[column] = (data_copy[column] - mean_val) / std_val
    return data_copy

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names (default: all numeric columns)
        outlier_factor: IQR multiplier for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_factor)
            cleaned_data = standardize_zscore(cleaned_data, column)
    
    return cleaned_data

def get_statistics(data, column):
    """
    Get statistical summary for a column.
    
    Args:
        data: pandas DataFrame
        column: column name
    
    Returns:
        Dictionary with statistical measures
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    return {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'q1': data[column].quantile(0.25),
        'q3': data[column].quantile(0.75),
        'count': data[column].count(),
        'missing': data[column].isnull().sum()
    }