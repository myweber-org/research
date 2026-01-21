
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list, optional): Columns to consider for duplicates.
    keep (str, optional): Which duplicates to keep.
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    if subset is not None:
        missing_cols = [col for col in subset if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep, ignore_index=True)
    return cleaned_df

def validate_dataframe(df):
    """
    Perform basic validation on DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    return validation_results

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to clean.
    
    Returns:
    pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 3, 2, 4, 1],
        'name': ['Alice', 'Bob', 'Charlie', 'Bob', 'David', 'Alice'],
        'age': [25, 30, 35, 30, 40, 25],
        'score': [85.5, 92.0, 78.5, 92.0, 88.0, 85.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_duplicates(df, subset=['id', 'name'])
    print("DataFrame after removing duplicates:")
    print(cleaned_df)
    print()
    
    validation = validate_dataframe(cleaned_df)
    print("Validation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()