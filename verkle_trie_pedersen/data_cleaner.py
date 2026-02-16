
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or np.array): The dataset
    column (int): Index of the column to clean
    
    Returns:
    np.array: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    col_data = data[:, column].astype(float)
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (np.array): The dataset
    column (int): Index of the column
    
    Returns:
    dict: Dictionary containing statistics
    """
    col_data = data[:, column].astype(float)
    
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data),
        'min': np.min(col_data),
        'max': np.max(col_data)
    }
    
    return stats

if __name__ == "__main__":
    sample_data = np.array([
        [1, 150.5, 'A'],
        [2, 165.3, 'B'],
        [3, 172.8, 'A'],
        [4, 158.1, 'C'],
        [5, 210.7, 'B'],
        [6, 95.2, 'A'],
        [7, 168.9, 'C'],
        [8, 155.4, 'B'],
        [9, 190.3, 'A'],
        [10, 82.1, 'C']
    ])
    
    print("Original data shape:", sample_data.shape)
    
    cleaned_data = remove_outliers_iqr(sample_data, 1)
    print("Cleaned data shape:", cleaned_data.shape)
    
    stats = calculate_statistics(cleaned_data, 1)
    print("Statistics after cleaning:", stats)