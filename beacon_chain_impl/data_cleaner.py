
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    column (int): Index of the column to clean.
    
    Returns:
    numpy.ndarray: Data with outliers removed.
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

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    column (int): Index of the column to analyze.
    
    Returns:
    dict: Dictionary containing statistics.
    """
    cleaned_data = remove_outliers_iqr(data, column)
    col_data = cleaned_data[:, column]
    
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data),
        'min': np.min(col_data),
        'max': np.max(col_data),
        'count': len(col_data)
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    sample_data = np.random.randn(1000, 3)
    sample_data[:, 1] = sample_data[:, 1] * 10 + 5  # Make second column have different scale
    
    print("Original data shape:", sample_data.shape)
    
    cleaned = remove_outliers_iqr(sample_data, 1)
    print("Cleaned data shape:", cleaned.shape)
    
    stats = calculate_statistics(sample_data, 1)
    print("Statistics for column 1:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")