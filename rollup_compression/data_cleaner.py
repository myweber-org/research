
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
        columns (list): Specific columns to clean, None for all columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns
    
    for col in columns:
        if df_clean[col].isnull().any():
            if strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
    
    return df_clean

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & 
                               (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_data(df, columns=None):
    """
    Standardize numerical columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to standardize
    
    Returns:
        pd.DataFrame: Standardized DataFrame
    """
    df_standardized = df.copy()
    
    if columns is None:
        columns = df_standardized.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_standardized.columns:
            mean_val = df_standardized[col].mean()
            std_val = df_standardized[col].std()
            if std_val > 0:
                df_standardized[col] = (df_standardized[col] - mean_val) / std_val
    
    return df_standardized

def normalize_data(df, columns=None, range_min=0, range_max=1):
    """
    Normalize numerical columns to specified range.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to normalize
        range_min (float): Minimum value after normalization
        range_max (float): Maximum value after normalization
    
    Returns:
        pd.DataFrame: Normalized DataFrame
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df_normalized.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_normalized.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val > min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                df_normalized[col] = df_normalized[col] * (range_max - range_min) + range_min
    
    return df_normalized

def clean_dataset(df, missing_strategy='mean', remove_outliers=True, 
                  standardize=False, normalize=False):
    """
    Complete data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values
        remove_outliers (bool): Whether to remove outliers
        standardize (bool): Whether to standardize data
        normalize (bool): Whether to normalize data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = clean_missing_values(df, strategy=missing_strategy)
    
    if remove_outliers:
        df_clean = remove_outliers_iqr(df_clean)
    
    if standardize:
        df_clean = standardize_data(df_clean)
    
    if normalize:
        df_clean = normalize_data(df_clean)
    
    return df_clean

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 60],
        'C': [100, 200, 300, 400, 500, 600]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, missing_strategy='mean', remove_outliers=True)
    print("\nCleaned DataFrame:")
    print(cleaned_df)