
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range method.
    
    Args:
        data: numpy array or list of numerical values
        column: index or key of the column to process (if data is 2D)
    
    Returns:
        Cleaned data with outliers removed
    """
    if isinstance(data, np.ndarray) and data.ndim == 2:
        column_data = data[:, column]
    else:
        column_data = np.array(data)
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if isinstance(data, np.ndarray) and data.ndim == 2:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        return data[mask]
    else:
        return column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Args:
        data: numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, std, min, and max
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }

def normalize_data(data, method='minmax'):
    """
    Normalize data using specified method.
    
    Args:
        data: numpy array of numerical values
        method: normalization method ('minmax' or 'zscore')
    
    Returns:
        Normalized data
    """
    if method == 'minmax':
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min == 0:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    elif method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def example_usage():
    """Example demonstrating the data cleaning functions."""
    np.random.seed(42)
    
    # Generate sample data with outliers
    sample_data = np.random.normal(100, 15, 1000)
    sample_data = np.append(sample_data, [500, 600, -200])  # Add outliers
    
    print("Original data statistics:")
    stats = calculate_statistics(sample_data)
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    # Remove outliers
    cleaned_data = remove_outliers_iqr(sample_data, 0)
    
    print("\nCleaned data statistics:")
    cleaned_stats = calculate_statistics(cleaned_data)
    for key, value in cleaned_stats.items():
        print(f"{key}: {value:.2f}")
    
    # Normalize the cleaned data
    normalized_data = normalize_data(cleaned_data, method='zscore')
    
    print(f"\nOriginal data points: {len(sample_data)}")
    print(f"Cleaned data points: {len(cleaned_data)}")
    print(f"Outliers removed: {len(sample_data) - len(cleaned_data)}")
    
    return cleaned_data, normalized_data

if __name__ == "__main__":
    cleaned, normalized = example_usage()
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, outlier_method='iqr', normalize_method='minmax', outlier_columns=None, normalize_columns=None):
    """
    Comprehensive data cleaning pipeline
    """
    if outlier_columns is None:
        outlier_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if normalize_columns is None:
        normalize_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    removal_stats = {}
    
    for column in outlier_columns:
        if column in data.columns and pd.api.types.is_numeric_dtype(data[column]):
            if outlier_method == 'iqr':
                cleaned_data, removed = remove_outliers_iqr(cleaned_data, column)
            elif outlier_method == 'zscore':
                cleaned_data, removed = remove_outliers_zscore(cleaned_data, column)
            else:
                raise ValueError(f"Unknown outlier method: {outlier_method}")
            
            removal_stats[column] = removed
    
    for column in normalize_columns:
        if column in cleaned_data.columns and pd.api.types.is_numeric_dtype(cleaned_data[column]):
            if normalize_method == 'minmax':
                cleaned_data[column] = normalize_minmax(cleaned_data, column)
            elif normalize_method == 'zscore':
                cleaned_data[column] = normalize_zscore(cleaned_data, column)
            else:
                raise ValueError(f"Unknown normalize method: {normalize_method}")
    
    return cleaned_data, removal_stats

def validate_data(data, required_columns=None, check_missing=True, check_duplicates=True):
    """
    Validate dataset for common issues
    """
    validation_results = {
        'is_valid': True,
        'issues': []
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_columns}")
    
    if check_missing:
        missing_values = data.isnull().sum().sum()
        if missing_values > 0:
            validation_results['issues'].append(f"Found {missing_values} missing values")
    
    if check_duplicates:
        duplicate_rows = data.duplicated().sum()
        if duplicate_rows > 0:
            validation_results['issues'].append(f"Found {duplicate_rows} duplicate rows")
    
    return validation_results