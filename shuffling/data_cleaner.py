
import pandas as pd
import numpy as np

def clean_dataframe(df, text_columns=None, fill_na_value=0):
    """
    Clean a pandas DataFrame by removing null values and standardizing text columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    text_columns (list): List of column names containing text data
    fill_na_value: Value to fill NaN entries with (default: 0)
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Fill missing values
    cleaned_df = cleaned_df.fillna(fill_na_value)
    
    # Standardize text columns if specified
    if text_columns:
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def sample_data_cleaning():
    """
    Example usage of the data cleaning functions.
    """
    # Create sample data
    data = {
        'name': ['Alice', 'Bob', None, 'Charlie', 'Alice'],
        'age': [25, 30, 35, None, 25],
        'city': ['New York', 'los angeles', 'Chicago', 'CHICAGO', 'New York']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataframe(df, text_columns=['name', 'city'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    # Validate the data
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['name', 'age'])
    print(f"Validation result: {is_valid}")
    print(f"Validation message: {message}")
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = sample_data_cleaning()