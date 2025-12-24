
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