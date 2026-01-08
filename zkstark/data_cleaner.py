import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Index of column to clean
    
    Returns:
    numpy.ndarray: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    column_data = data[:, column]
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Index of column to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    cleaned_data = remove_outliers_iqr(data, column)
    column_cleaned = cleaned_data[:, column]
    
    stats = {
        'mean': np.mean(column_cleaned),
        'median': np.median(column_cleaned),
        'std': np.std(column_cleaned),
        'min': np.min(column_cleaned),
        'max': np.max(column_cleaned),
        'original_size': len(data),
        'cleaned_size': len(cleaned_data),
        'removed_count': len(data) - len(cleaned_data)
    }
    
    return stats

def example_usage():
    """
    Example demonstrating the usage of data cleaning functions.
    """
    np.random.seed(42)
    
    sample_data = np.random.randn(1000, 3)
    sample_data[:, 1] = sample_data[:, 1] * 10 + 50
    
    outliers = np.random.randn(50, 3) * 20 + 100
    sample_data = np.vstack([sample_data, outliers])
    
    print(f"Original data shape: {sample_data.shape}")
    
    cleaned = remove_outliers_iqr(sample_data, 1)
    print(f"Cleaned data shape: {cleaned.shape}")
    
    stats = calculate_statistics(sample_data, 1)
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    example_usage()