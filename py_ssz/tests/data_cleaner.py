
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data (np.ndarray): Input data array
        column (int): Column index to process
    
    Returns:
        np.ndarray: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    column_data = data[:, column]
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Args:
        data (np.ndarray): Input data array
        column (int): Column index to analyze
    
    Returns:
        dict: Dictionary containing statistical measures
    """
    cleaned_data = remove_outliers_iqr(data, column)
    column_data = cleaned_data[:, column]
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data),
        'count': len(column_data)
    }
    
    return stats

def validate_data_shape(data, expected_columns):
    """
    Validate that data has the expected number of columns.
    
    Args:
        data (np.ndarray): Input data array
        expected_columns (int): Expected number of columns
    
    Returns:
        bool: True if shape is valid, False otherwise
    """
    if len(data.shape) != 2:
        return False
    
    return data.shape[1] == expected_columns

def example_usage():
    """
    Demonstrate usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    sample_data = np.random.randn(100, 3)
    sample_data[:, 1] = sample_data[:, 1] * 10 + 5
    
    print("Original data shape:", sample_data.shape)
    
    cleaned_data = remove_outliers_iqr(sample_data, 1)
    print("Cleaned data shape:", cleaned_data.shape)
    
    stats = calculate_statistics(sample_data, 1)
    print("Statistics for column 1:", stats)
    
    is_valid = validate_data_shape(sample_data, 3)
    print("Data shape validation:", is_valid)

if __name__ == "__main__":
    example_usage()