import pandas as pd
import numpy as np

def clean_missing_data(file_path, strategy='mean', columns=None):
    """
    Clean missing data in a CSV file using specified strategy.
    
    Args:
        file_path (str): Path to the CSV file
        strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        columns (list): List of columns to apply cleaning to (None for all columns)
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(file_path)
        
        if columns is None:
            columns = df.columns
        
        for col in columns:
            if col in df.columns:
                if df[col].isnull().any():
                    if strategy == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif strategy == 'median':
                        df[col].fillna(df[col].median(), inplace=True)
                    elif strategy == 'mode':
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif strategy == 'drop':
                        df.dropna(subset=[col], inplace=True)
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def detect_outliers(df, column, method='iqr'):
    """
    Detect outliers in a specified column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to check for outliers
        method (str): Method for outlier detection ('iqr' or 'zscore')
    
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    if column not in df.columns:
        return pd.Series([False] * len(df))
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        z_scores = (df[column] - mean) / std
        return np.abs(z_scores) > 3
    
    return pd.Series([False] * len(df))

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        output_path (str): Path to save the cleaned data
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving cleaned data: {str(e)}")
        return False