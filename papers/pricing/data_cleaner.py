
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range (IQR) method.
    
    Args:
        data (np.ndarray): Input data array.
        column (int): Column index to check for outliers.
    
    Returns:
        np.ndarray: Data with outliers removed.
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
    Calculate basic statistics for the cleaned data.
    
    Args:
        data (np.ndarray): Input data array.
    
    Returns:
        dict: Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': np.mean(data, axis=0),
        'median': np.median(data, axis=0),
        'std': np.std(data, axis=0)
    }
    return stats

def clean_dataset(data, outlier_columns=None):
    """
    Main function to clean dataset by removing outliers from specified columns.
    
    Args:
        data (np.ndarray): Input data array.
        outlier_columns (list): List of column indices to check for outliers.
    
    Returns:
        tuple: Cleaned data and statistics dictionary.
    """
    if outlier_columns is None:
        outlier_columns = list(range(data.shape[1]))
    
    cleaned_data = data.copy()
    for col in outlier_columns:
        cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    stats = calculate_statistics(cleaned_data)
    return cleaned_data, stats

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = np.random.randn(100, 3) * 10 + 50
    
    print("Original data shape:", sample_data.shape)
    cleaned, statistics = clean_dataset(sample_data, [0, 1, 2])
    print("Cleaned data shape:", cleaned.shape)
    
    for key, value in statistics.items():
        print(f"{key}: {value}")