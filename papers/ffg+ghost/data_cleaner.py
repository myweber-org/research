import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or array-like): The dataset containing the column to clean.
    column (int or str): The index or name of the column to process.
    
    Returns:
    tuple: (cleaned_data, removed_indices) where cleaned_data is the data with outliers removed,
           and removed_indices are the indices of removed outliers.
    """
    if isinstance(data, list):
        data_array = np.array(data)
    else:
        data_array = data.copy()
    
    column_data = data_array[:, column] if isinstance(column, int) else data_array[column]
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    cleaned_data = data_array[outlier_mask]
    removed_indices = np.where(~outlier_mask)[0]
    
    return cleaned_data, removed_indices

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (array-like): The dataset.
    column (int or str): The column to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, std, min, and max.
    """
    column_data = data[:, column] if isinstance(column, int) else data[column]
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data)
    }
    
    return stats

def normalize_column(data, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Parameters:
    data (array-like): The dataset.
    column (int or str): The column to normalize.
    method (str): Normalization method ('minmax' or 'zscore').
    
    Returns:
    array-like: Normalized column data.
    """
    column_data = data[:, column] if isinstance(column, int) else data[column]
    
    if method == 'minmax':
        min_val = np.min(column_data)
        max_val = np.max(column_data)
        if max_val - min_val == 0:
            return np.zeros_like(column_data)
        normalized = (column_data - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = np.mean(column_data)
        std_val = np.std(column_data)
        if std_val == 0:
            return np.zeros_like(column_data)
        normalized = (column_data - mean_val) / std_val
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return normalized

if __name__ == "__main__":
    # Example usage
    sample_data = np.array([
        [1, 150, 25],
        [2, 160, 30],
        [3, 170, 22],
        [4, 180, 35],
        [5, 190, 28],
        [6, 200, 100],  # Outlier
        [7, 210, 32],
        [8, 220, 29],
        [9, 230, 31],
        [10, 240, 27]
    ])
    
    print("Original data shape:", sample_data.shape)
    
    # Remove outliers from column 2 (age column)
    cleaned_data, removed_idx = remove_outliers_iqr(sample_data, 2)
    print("Cleaned data shape:", cleaned_data.shape)
    print("Removed indices:", removed_idx)
    
    # Calculate statistics
    stats = calculate_statistics(sample_data, 1)
    print("Height statistics:", stats)
    
    # Normalize height column
    normalized_height = normalize_column(sample_data, 1, 'minmax')
    print("Normalized height (first 5):", normalized_height[:5])