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
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing statistical measures.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    data = {'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 102, 12, 14, 13, 12, 11, 14, 13, 12, 11, 100]}
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    stats = calculate_basic_stats(cleaned_df, 'values')
    print("\nBasic Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # If specific columns are provided, clean only those; otherwise, clean all object columns
    if columns_to_clean is None:
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns
    
    for col in columns_to_clean:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
    
    return cleaned_df

def _normalize_string(s):
    """
    Normalize a string: lowercase, strip whitespace, and remove extra spaces.
    """
    if isinstance(s, str):
        s = s.lower()
        s = s.strip()
        s = re.sub(r'\s+', ' ', s)
    return s

def validate_email(email):
    """
    Simple email validation using regex.
    """
    if not isinstance(email, str):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_method='iqr', threshold=1.5):
    """
    Clean dataset by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for outlier detection ('iqr', 'zscore')
    threshold (float): Threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    
    df_clean = df.copy()
    
    # Handle missing values
    if strategy == 'mean':
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    elif strategy == 'median':
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    elif strategy == 'mode':
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    elif strategy == 'drop':
        df_clean = df_clean.dropna()
    
    # Handle outliers for numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Cap outliers
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            
        elif outlier_method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores < threshold]
    
    return df_clean

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    subset (list): Columns to consider for duplicates
    keep (str): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: Dataframe without duplicates
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_data(df, method='minmax'):
    """
    Normalize numeric columns in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    method (str): Normalization method ('minmax', 'standard')
    
    Returns:
    pd.DataFrame: Normalized dataframe
    """
    df_norm = df.copy()
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
    
    if method == 'minmax':
        for col in numeric_cols:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    
    elif method == 'standard':
        for col in numeric_cols:
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val > 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'x', 'y', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, strategy='mean', outlier_method='iqr')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    normalized_df = normalize_data(cleaned_df, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized_df)