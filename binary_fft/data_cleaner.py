import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Columns to consider for duplicates
    keep (str): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    if subset is None:
        subset = df.columns.tolist()
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    print(f"Removed {len(df) - len(cleaned_df)} duplicate rows")
    print(f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df

def clean_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): 'drop' to remove rows, 'fill' to fill values
    fill_value: Value to fill missing values with
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        return df
    
    if strategy == 'drop':
        cleaned_df = df.dropna()
        print(f"Removed {len(df) - len(cleaned_df)} rows with missing values")
    elif strategy == 'fill':
        if fill_value is None:
            fill_value = df.mean(numeric_only=True)
        cleaned_df = df.fillna(fill_value)
        print("Filled missing values")
    else:
        cleaned_df = df.copy()
        print("No missing value handling performed")
    
    return cleaned_df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    required_columns (list): Columns that must be present
    numeric_columns (list): Columns that should be numeric
    
    Returns:
    bool: True if validation passes
    """
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if numeric_columns:
        non_numeric = []
        for col in numeric_columns:
            if col in df.columns:
                if not np.issubdtype(df[col].dtype, np.number):
                    non_numeric.append(col)
        
        if non_numeric:
            print(f"Non-numeric columns found: {non_numeric}")
            return False
    
    print("Data validation passed")
    return True

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 3, 1, 4, 5, 3],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David', 'Eve', 'Charlie'],
        'age': [25, 30, 35, 25, 28, np.nan, 35],
        'score': [85.5, 92.0, 78.5, 85.5, 88.0, 95.0, 78.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    df_clean = remove_duplicates(df, subset=['id', 'name'])
    print("\n" + "="*50 + "\n")
    
    df_clean = clean_missing_values(df_clean, strategy='fill', fill_value=0)
    print("\n" + "="*50 + "\n")
    
    is_valid = validate_data(df_clean, 
                           required_columns=['id', 'name', 'age', 'score'],
                           numeric_columns=['age', 'score'])
    
    print("\nCleaned DataFrame:")
    print(df_clean)

if __name__ == "__main__":
    main()