
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or np.array): Input data
    column (int): Index of column to process (if 2D array) or ignored (if 1D array)
    
    Returns:
    np.array: Data with outliers removed
    """
    data_array = np.array(data)
    
    if data_array.ndim == 2:
        column_data = data_array[:, column]
    else:
        column_data = data_array
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if data_array.ndim == 2:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        return data_array[mask]
    else:
        return column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (list or np.array): Input data
    
    Returns:
    dict: Dictionary containing mean, median, std, min, max
    """
    data_array = np.array(data)
    
    return {
        'mean': np.mean(data_array),
        'median': np.median(data_array),
        'std': np.std(data_array),
        'min': np.min(data_array),
        'max': np.max(data_array)
    }

if __name__ == "__main__":
    sample_data = [10, 12, 13, 15, 16, 18, 20, 22, 24, 100]
    
    cleaned_data = remove_outliers_iqr(sample_data, 0)
    stats = calculate_statistics(cleaned_data)
    
    print(f"Original data: {sample_data}")
    print(f"Cleaned data: {cleaned_data}")
    print(f"Statistics: {stats}")