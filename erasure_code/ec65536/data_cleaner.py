
import pandas as pd
import re

def clean_string_column(series, case='lower', strip=True, remove_special=True):
    """
    Standardize string values in a pandas Series.
    """
    if not pd.api.types.is_string_dtype(series):
        series = series.astype(str)
    
    if case == 'lower':
        series = series.str.lower()
    elif case == 'upper':
        series = series.str.upper()
    
    if strip:
        series = series.str.strip()
    
    if remove_special:
        series = series.apply(lambda x: re.sub(r'[^\w\s]', '', x))
    
    return series

def remove_duplicate_rows(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame with optional subset.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def standardize_numeric(series, fillna=0):
    """
    Convert series to numeric type and fill missing values.
    """
    series = pd.to_numeric(series, errors='coerce')
    return series.fillna(fillna)

def clean_dataframe(df, string_columns=None, numeric_columns=None):
    """
    Apply cleaning operations to multiple columns.
    """
    df_clean = df.copy()
    
    if string_columns:
        for col in string_columns:
            if col in df_clean.columns:
                df_clean[col] = clean_string_column(df_clean[col])
    
    if numeric_columns:
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = standardize_numeric(df_clean[col])
    
    return df_clean
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
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

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.randn(100) * 10 + 50,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    print(f"Original dataset shape: {df.shape}")
    
    cleaned_df = clean_dataset(df, ['value'])
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    
    # Save cleaned data
    cleaned_df.to_csv('cleaned_data.csv', index=False)
    print("Cleaned data saved to 'cleaned_data.csv'")