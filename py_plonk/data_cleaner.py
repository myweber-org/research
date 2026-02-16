
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data (np.ndarray): Input data array
        column (int): Index of column to clean
    
    Returns:
        np.ndarray: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Args:
        data (np.ndarray): Input data array
    
    Returns:
        dict: Dictionary containing mean, median, and std
    """
    stats = {
        'mean': np.mean(data, axis=0),
        'median': np.median(data, axis=0),
        'std': np.std(data, axis=0)
    }
    return stats

def normalize_data(data):
    """
    Normalize data using min-max scaling.
    
    Args:
        data (np.ndarray): Input data array
    
    Returns:
        np.ndarray: Normalized data
    """
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    
    # Avoid division by zero
    range_vals = data_max - data_min
    range_vals[range_vals == 0] = 1
    
    normalized = (data - data_min) / range_vals
    return normalized

if __name__ == "__main__":
    # Example usage
    sample_data = np.random.randn(100, 3) * 10 + 5
    print("Original shape:", sample_data.shape)
    
    cleaned_data = remove_outliers_iqr(sample_data, 0)
    print("Cleaned shape:", cleaned_data.shape)
    
    stats = calculate_statistics(cleaned_data)
    print("Statistics:", stats)
    
    normalized_data = normalize_data(cleaned_data)
    print("Normalized data range:", np.min(normalized_data), np.max(normalized_data))