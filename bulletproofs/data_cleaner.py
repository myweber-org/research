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
    Returns boolean Series where True indicates an outlier.
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
    Standardize a column to have mean=0 and std=1.
    """
    if column in df.columns:
        df_standardized = df.copy()
        mean_val = df[column].mean()
        std_val = df[column].std()
        
        if std_val > 0:
            df_standardized[column] = (df[column] - mean_val) / std_val
        else:
            df_standardized[column] = 0
            
        return df_standardized
    return df

def clean_dataset(df, missing_strategy='remove', outlier_columns=None):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    - df: Input DataFrame
    - missing_strategy: 'remove' or 'mean'
    - outlier_columns: List of columns to check for outliers
    
    Returns:
    - Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df)
    elif missing_strategy == 'mean':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        cleaned_df = fill_missing_with_mean(cleaned_df, numeric_cols)
    
    # Remove outliers
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers(cleaned_df, col)
    
    return cleaned_df

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
    
    cleaned = clean_dataset(df, missing_strategy='mean', outlier_columns=['A'])
    print("\nCleaned DataFrame:")
    print(cleaned)