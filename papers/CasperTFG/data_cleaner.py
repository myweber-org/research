
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or np.array): The dataset containing the column to clean.
    column (int): Index of the column to process.
    
    Returns:
    np.array: Data with outliers removed from the specified column.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    column_data = data[:, column].astype(float)
    
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
    
    Parameters:
    data (np.array): The cleaned dataset.
    column (int): Index of the column to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    column_data = data[:, column].astype(float)
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std_dev': np.std(column_data),
        'count': len(column_data)
    }
    
    return stats

def clean_dataset(data, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Parameters:
    data (list or np.array): The original dataset.
    columns_to_clean (list): List of column indices to clean.
    
    Returns:
    tuple: (cleaned_data, removal_stats) where removal_stats is a dictionary
           showing how many values were removed from each column.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    cleaned_data = data.copy()
    removal_stats = {}
    
    for column in columns_to_clean:
        original_count = len(cleaned_data)
        cleaned_data = remove_outliers_iqr(cleaned_data, column)
        removed_count = original_count - len(cleaned_data)
        removal_stats[column] = removed_count
    
    return cleaned_data, removal_stats

if __name__ == "__main__":
    # Example usage
    sample_data = np.array([
        [1, 10.5, 100],
        [2, 12.3, 150],
        [3, 9.8, 120],
        [4, 25.6, 80],    # Potential outlier in column 1
        [5, 11.2, 130],
        [6, 8.9, 110],
        [7, 30.1, 90],    # Potential outlier in column 1
        [8, 10.8, 140]
    ])
    
    print("Original data shape:", sample_data.shape)
    
    cleaned_data, stats = clean_dataset(sample_data, [1])
    
    print("Cleaned data shape:", cleaned_data.shape)
    print("Values removed per column:", stats)
    
    column_stats = calculate_statistics(cleaned_data, 1)
    print("Statistics for cleaned column 1:", column_stats)