
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")

def normalize_data(df, columns):
    result = df.copy()
    for col in columns:
        if col in df.columns:
            result[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return result

def process_dataset(df, numeric_columns):
    df_clean = clean_missing_values(df, strategy='median')
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean = remove_outliers_iqr(df_clean, col)
    return normalize_data(df_clean, numeric_columns)
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
    if isinstance(data, list):
        data_array = np.array(data)
    else:
        data_array = data
    
    if isinstance(column, str):
        raise ValueError("Column name handling requires pandas DataFrame. Use integer index for list/array data.")
    
    column_data = data_array[:, column].astype(float)
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (column_data < lower_bound) | (column_data > upper_bound)
    clean_mask = ~outlier_mask
    
    cleaned_data = data_array[clean_mask].tolist()
    outlier_indices = np.where(outlier_mask)[0].tolist()
    
    return cleaned_data, outlier_indices

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (list or array-like): The dataset.
    column (int): The index of the column to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, std, min, and max.
    """
    if isinstance(data, list):
        data_array = np.array(data)
    else:
        data_array = data
    
    column_data = data_array[:, column].astype(float)
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data)
    }
    
    return stats

def example_usage():
    """
    Example demonstrating the usage of data cleaning functions.
    """
    sample_data = [
        [1, 150.5],
        [2, 155.2],
        [3, 160.1],
        [4, 162.3],
        [5, 158.7],
        [6, 50.0],    # Potential outlier (low)
        [7, 165.4],
        [8, 168.9],
        [9, 170.2],
        [10, 250.0]   # Potential outlier (high)
    ]
    
    print("Original data:")
    for row in sample_data:
        print(row)
    
    cleaned_data, outliers = remove_outliers_iqr(sample_data, column=1)
    
    print(f"\nRemoved {len(outliers)} outliers at indices: {outliers}")
    print("\nCleaned data:")
    for row in cleaned_data:
        print(row)
    
    stats = calculate_statistics(cleaned_data, column=1)
    print(f"\nStatistics after cleaning:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    example_usage()