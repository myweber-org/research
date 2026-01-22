
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
    
    if subset is not None:
        if not all(col in df.columns for col in subset):
            raise ValueError("All subset columns must exist in DataFrame")
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df.reset_index(drop=True)

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
        removed_count = len(df) - len(cleaned_df)
        print(f"Removed {removed_count} rows with missing values")
    elif strategy == 'fill':
        if fill_value is None:
            fill_value = df.select_dtypes(include=[np.number]).mean()
        cleaned_df = df.fillna(fill_value)
        print(f"Filled missing values with {fill_value}")
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"DataFrame validated: {len(df)} rows, {len(df.columns)} columns")
    return True

def clean_data_pipeline(df, cleaning_steps):
    """
    Execute multiple cleaning steps in sequence.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    cleaning_steps (list): List of cleaning function configurations
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    current_df = df.copy()
    
    for step in cleaning_steps:
        func = step['function']
        params = step.get('params', {})
        current_df = func(current_df, **params)
    
    return current_df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 2, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Bob', 'David', 'Eve', 'Eve'],
        'score': [85, 90, None, 90, 75, 88, 88],
        'department': ['HR', 'IT', 'IT', 'IT', 'HR', 'Finance', 'Finance']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50)
    
    cleaned_df = remove_duplicates(df, subset=['id', 'name'], keep='first')
    cleaned_df = clean_missing_values(cleaned_df, strategy='fill', fill_value=0)
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)