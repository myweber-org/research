
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or array-like): The dataset containing the column to clean.
    column (int or str): The index or name of the column to process.
    
    Returns:
    tuple: A tuple containing:
        - cleaned_data (list): Data with outliers removed.
        - outlier_indices (list): Indices of removed outliers.
    """
    if not data:
        return [], []
    
    # Convert data to numpy array for easier calculations
    data_array = np.array(data)
    
    # Extract the column values
    if isinstance(column, int):
        column_values = data_array[:, column].astype(float)
    else:
        # If column is a string, we'd need a more complex implementation
        # For simplicity, assuming integer index for this example
        column_values = data_array[:, int(column)].astype(float)
    
    # Calculate Q1, Q3 and IQR
    Q1 = np.percentile(column_values, 25)
    Q3 = np.percentile(column_values, 75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify non-outliers
    non_outlier_mask = (column_values >= lower_bound) & (column_values <= upper_bound)
    outlier_indices = np.where(~non_outlier_mask)[0].tolist()
    
    # Filter data to remove outliers
    cleaned_data = [data[i] for i in range(len(data)) if non_outlier_mask[i]]
    
    return cleaned_data, outlier_indices

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (list or array-like): The dataset.
    column (int): The index of the column to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, std, min, and max.
    """
    if not data:
        return {}
    
    data_array = np.array(data)
    column_values = data_array[:, column].astype(float)
    
    stats = {
        'mean': np.mean(column_values),
        'median': np.median(column_values),
        'std': np.std(column_values),
        'min': np.min(column_values),
        'max': np.max(column_values),
        'count': len(column_values)
    }
    
    return stats

# Example usage
if __name__ == "__main__":
    # Sample data
    sample_data = [
        [1, 10.5],
        [2, 12.3],
        [3, 11.8],
        [4, 100.0],  # This is an outlier
        [5, 10.9],
        [6, 11.2],
        [7, 9.8],
        [8, 12.1],
        [9, 200.0],  # This is an outlier
        [10, 11.5]
    ]
    
    print("Original data:")
    for row in sample_data:
        print(row)
    
    # Remove outliers from column 1
    cleaned_data, outliers = remove_outliers_iqr(sample_data, 1)
    
    print(f"\nRemoved {len(outliers)} outliers at indices: {outliers}")
    print("\nCleaned data:")
    for row in cleaned_data:
        print(row)
    
    # Calculate statistics
    stats = calculate_basic_stats(cleaned_data, 1)
    print(f"\nStatistics for cleaned data column 1:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")