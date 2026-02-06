
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        data (list or np.array): Input data
        column (int): Column index if data is 2D, ignored if data is 1D
    
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
    
    Args:
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

def clean_dataset(data, columns_to_clean=None):
    """
    Clean dataset by removing outliers from specified columns.
    
    Args:
        data (np.array): 2D array of data
        columns_to_clean (list): List of column indices to clean
    
    Returns:
        np.array: Cleaned dataset
    """
    if columns_to_clean is None:
        columns_to_clean = list(range(data.shape[1]))
    
    cleaned_data = data.copy()
    
    for col in columns_to_clean:
        if col < data.shape[1]:
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    return cleaned_data

if __name__ == "__main__":
    sample_data = np.random.randn(100, 3) * 10 + 50
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics:", calculate_statistics(sample_data[:, 0]))
    
    cleaned = clean_dataset(sample_data, columns_to_clean=[0, 1, 2])
    
    print("Cleaned data shape:", cleaned.shape)
    print("Cleaned statistics:", calculate_statistics(cleaned[:, 0]))