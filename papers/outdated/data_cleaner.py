
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    multiplier (float): IQR multiplier
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized column values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.Series: Standardized column values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_multiplier=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names
    outlier_multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            # Remove outliers
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_multiplier)
            
            # Normalize the column
            cleaned_df[f"{col}_normalized"] = normalize_minmax(cleaned_df, col)
            cleaned_df[f"{col}_standardized"] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def get_summary_statistics(df, numeric_columns):
    """
    Get summary statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    summary = {}
    
    for col in numeric_columns:
        if col in df.columns:
            summary[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q1': df[col].quantile(0.25),
                'q3': df[col].quantile(0.75),
                'count': df[col].count(),
                'missing': df[col].isnull().sum()
            }
    
    return pd.DataFrame(summary).T

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'temperature': [22, 25, 28, 31, 34, 100, 23, 26, 29, 32],
        'humidity': [45, 50, 55, 60, 65, 200, 47, 52, 57, 62],
        'pressure': [1013, 1015, 1012, 1014, 1016, 1500, 1013, 1015, 1012, 1014]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original Data:")
    print(df)
    print("\nSummary Statistics:")
    print(get_summary_statistics(df, ['temperature', 'humidity', 'pressure']))
    
    cleaned_df = clean_dataset(df, ['temperature', 'humidity', 'pressure'])
    print("\nCleaned Data:")
    print(cleaned_df)