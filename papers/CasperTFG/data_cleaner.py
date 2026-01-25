
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
import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    If columns specified, only consider missing values in those columns.
    """
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def fill_missing_with_mean(df, columns):
    """
    Fill missing values in specified columns with column mean.
    """
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            df_filled[col] = df[col].fillna(mean_val)
    return df_filled

def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method for a specific column.
    Returns boolean Series indicating outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def remove_outliers(df, column):
    """
    Remove rows where specified column contains outliers (IQR method).
    """
    outliers = detect_outliers_iqr(df, column)
    return df[~outliers]

def standardize_column(df, column):
    """
    Standardize a column using z-score normalization.
    """
    if column in df.columns:
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df[column] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(df, missing_strategy='remove', outlier_columns=None):
    """
    Comprehensive data cleaning function.
    missing_strategy: 'remove' or 'mean'
    outlier_columns: list of columns to remove outliers from
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df)
    elif missing_strategy == 'mean':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df = fill_missing_with_mean(cleaned_df, numeric_cols)
    
    # Remove outliers
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers(cleaned_df, col)
    
    return cleaned_df