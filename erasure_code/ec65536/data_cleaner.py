
import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and optionally dropping columns.
    
    Parameters:
    filepath (str): Path to the CSV file.
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop').
    columns_to_drop (list): List of column names to drop from the dataset.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {filepath}")
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif missing_strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif missing_strategy == 'mode':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
    elif missing_strategy == 'drop':
        df = df.dropna(subset=numeric_cols)
    else:
        raise ValueError("Invalid missing_strategy. Choose from 'mean', 'median', 'mode', 'drop'.")
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')
    
    df = df.reset_index(drop=True)
    
    return df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    column (str): Column name to process.
    multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
    pandas.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def standardize_columns(df, columns):
    """
    Standardize specified numeric columns to have zero mean and unit variance.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    columns (list): List of column names to standardize.
    
    Returns:
    pandas.DataFrame: DataFrame with standardized columns.
    """
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
        else:
            print(f"Warning: Column '{col}' is not numeric or does not exist. Skipping.")
    
    return df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, np.nan, 50],
        'C': ['X', 'Y', None, 'Z', 'X']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned = clean_csv_data('sample_data.csv', missing_strategy='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    
    cleaned_no_outliers = remove_outliers_iqr(cleaned, 'A')
    print("\nDataFrame after outlier removal:")
    print(cleaned_no_outliers)
    
    standardized = standardize_columns(cleaned, ['A', 'B'])
    print("\nDataFrame after standardization:")
    print(standardized)