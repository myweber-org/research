
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or array-like): The dataset.
    column (int or str): The column index or name to process.
    
    Returns:
    numpy.ndarray: Data with outliers removed.
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if isinstance(column, str):
        try:
            col_index = data.dtype.names.index(column)
            col_data = data[col_index]
        except (AttributeError, ValueError):
            raise ValueError("Column name not found in structured array")
    else:
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
    data (numpy.ndarray): The dataset.
    column (int): The column index.
    
    Returns:
    dict: Dictionary containing mean, median, and std.
    """
    col_data = data[:, column]
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data)
    }
    return stats

if __name__ == "__main__":
    sample_data = np.array([
        [1.2, 150],
        [2.3, 200],
        [3.1, 250],
        [100.5, 300],
        [4.2, 350],
        [5.7, 400],
        [6.8, 450],
        [7.9, 500]
    ])
    
    print("Original data shape:", sample_data.shape)
    cleaned_data = remove_outliers_iqr(sample_data, 0)
    print("Cleaned data shape:", cleaned_data.shape)
    
    if len(cleaned_data) > 0:
        stats = calculate_statistics(cleaned_data, 0)
        print("Statistics for column 0:", stats)