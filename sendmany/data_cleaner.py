import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or array): The dataset containing the column.
    column (int or str): The column index or name to process.
    
    Returns:
    tuple: (cleaned_data, outliers_removed)
    """
    if isinstance(data, list):
        data = np.array(data)
    
    col_data = data[:, column] if isinstance(column, int) else data[column]
    
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    cleaned_data = data[mask]
    outliers_removed = len(data) - len(cleaned_data)
    
    return cleaned_data, outliers_removed

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (array): The dataset.
    column (int or str): The column to analyze.
    
    Returns:
    dict: Dictionary containing statistics.
    """
    col_data = data[:, column] if isinstance(column, int) else data[column]
    
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data),
        'min': np.min(col_data),
        'max': np.max(col_data),
        'q1': np.percentile(col_data, 25),
        'q3': np.percentile(col_data, 75)
    }
    
    return stats

if __name__ == "__main__":
    sample_data = np.array([
        [1, 150],
        [2, 200],
        [3, 250],
        [4, 300],
        [5, 1000],
        [6, 50]
    ])
    
    print("Original data:")
    print(sample_data)
    
    cleaned, removed = remove_outliers_iqr(sample_data, 1)
    print(f"\nOutliers removed: {removed}")
    print("Cleaned data:")
    print(cleaned)
    
    stats = calculate_basic_stats(sample_data, 1)
    print("\nStatistics for column 1:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")