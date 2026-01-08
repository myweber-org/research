
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or np.array): Input data
    column (int or str): Column index or name if using pandas
    
    Returns:
    np.array: Data with outliers removed
    """
    if isinstance(data, list):
        data = np.array(data)
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    return filtered_data

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (np.array): Input data
    
    Returns:
    dict: Dictionary containing mean, median, std
    """
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'count': len(data)
    }
    
    return stats

def clean_dataset(data_array):
    """
    Main function to clean dataset by removing outliers.
    
    Parameters:
    data_array (np.array): 2D array of data
    
    Returns:
    tuple: (cleaned_data, removed_indices, statistics)
    """
    if len(data_array.shape) == 1:
        data_array = data_array.reshape(-1, 1)
    
    cleaned_data = []
    removed_indices = []
    
    for col in range(data_array.shape[1]):
        column_data = data_array[:, col]
        cleaned_column = remove_outliers_iqr(column_data, col)
        cleaned_data.append(cleaned_column)
        
        original_indices = np.arange(len(column_data))
        filtered_indices = original_indices[
            (column_data >= np.percentile(column_data, 25) - 1.5 * (np.percentile(column_data, 75) - np.percentile(column_data, 25))) &
            (column_data <= np.percentile(column_data, 75) + 1.5 * (np.percentile(column_data, 75) - np.percentile(column_data, 25)))
        ]
        removed = np.setdiff1d(original_indices, filtered_indices)
        removed_indices.append(removed)
    
    cleaned_data = np.column_stack(cleaned_data) if len(cleaned_data) > 1 else cleaned_data[0]
    
    stats = calculate_statistics(cleaned_data.flatten())
    
    return cleaned_data, removed_indices, stats

if __name__ == "__main__":
    # Example usage
    sample_data = np.random.randn(100, 3) * 10 + 50
    sample_data[0, 0] = 200  # Add an outlier
    
    cleaned, removed, statistics = clean_dataset(sample_data)
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Removed indices per column: {removed}")
    print(f"Statistics: {statistics}")