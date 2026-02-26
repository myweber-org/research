
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range (IQR) method.
    
    Args:
        data: pandas DataFrame containing the data
        column: string name of the column to process
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    outliers_removed = len(data) - len(filtered_data)
    print(f"Removed {outliers_removed} outliers from column '{column}'")
    
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column.
    
    Args:
        data: pandas DataFrame
        column: string name of the column
    
    Returns:
        Dictionary containing summary statistics
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'count': data[column].count()
    }
    
    return stats

def normalize_column(data, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        data: pandas DataFrame
        column: string name of the column to normalize
        method: string normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column added as new column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    if method == 'minmax':
        min_val = data[column].min()
        max_val = data[column].max()
        normalized = (data[column] - min_val) / (max_val - min_val)
        new_col_name = f"{column}_normalized_minmax"
    
    elif method == 'zscore':
        mean_val = data[column].mean()
        std_val = data[column].std()
        normalized = (data[column] - mean_val) / std_val
        new_col_name = f"{column}_normalized_zscore"
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    data[new_col_name] = normalized
    return data
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    data = {'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 13, 12, 10, 9, 8, 200]}
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    stats = calculate_summary_statistics(cleaned_df, 'values')
    print("\nSummary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")