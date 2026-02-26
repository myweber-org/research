
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def standardize_column(df, column):
    """
    Standardize a column to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to standardize.
    
    Returns:
    pd.DataFrame: DataFrame with standardized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = df[column].mean()
    std_val = df[column].std()
    
    if std_val == 0:
        return df
    
    df_copy = df.copy()
    df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    strategy (str): Imputation strategy ('mean', 'median', or 'drop').
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df.dropna(subset=numeric_cols)
    elif strategy == 'mean':
        impute_values = df[numeric_cols].mean()
    elif strategy == 'median':
        impute_values = df[numeric_cols].median()
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    df_filled = df.copy()
    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(impute_values)
    
    return df_filled

def clean_dataset(df, numeric_columns=None, outlier_removal=True, standardization=True, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_columns (list): List of numeric columns to process. If None, all numeric columns are used.
    outlier_removal (bool): Whether to remove outliers.
    standardization (bool): Whether to standardize columns.
    missing_strategy (str): Strategy for handling missing values.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if outlier_removal:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    if standardization:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = standardize_column(cleaned_df, col)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [5, np.nan, 15, 20, 25, 30, 35, 40, 45, 50]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataset(df, outlier_removal=True, standardization=True)
    print("Cleaned DataFrame:")
    print(cleaned)