import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing rows with null values
    and standardizing column names to lowercase with underscores.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Standardize column names
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')
    
    return df_cleaned

def validate_dataframe(df):
    """
    Validate that the DataFrame has no null values
    and column names follow the standardized format.
    """
    # Check for null values
    if df.isnull().any().any():
        return False, "DataFrame contains null values"
    
    # Check column name format
    for col in df.columns:
        if not isinstance(col, str):
            return False, f"Column name {col} is not a string"
        if ' ' in col or col != col.lower():
            return False, f"Column name {col} does not follow naming convention"
    
    return True, "DataFrame is valid"

def process_data(file_path):
    """
    Load data from a CSV file, clean it, and validate the result.
    """
    try:
        df = pd.read_csv(file_path)
        df_cleaned = clean_dataframe(df)
        is_valid, message = validate_dataframe(df_cleaned)
        
        if is_valid:
            print(f"Data cleaning successful: {message}")
            return df_cleaned
        else:
            print(f"Data cleaning failed: {message}")
            return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'Product Name': ['A', 'B', None, 'D'],
        'Price': [100, 200, 300, None],
        'Quantity': [10, 20, 30, 40]
    })
    
    cleaned = clean_dataframe(sample_data)
    print("Original DataFrame:")
    print(sample_data)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned)
    print(f"\nValidation result: {is_valid}, Message: {message}")
import pandas as pd
import re

def clean_dataframe(df, column_name):
    """
    Clean a specified column in a DataFrame by removing duplicates,
    stripping whitespace, and converting to lowercase.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df = df.drop_duplicates(subset=[column_name])
    df[column_name] = df[column_name].astype(str).str.strip().str.lower()
    df = df.reset_index(drop=True)
    return df

def remove_special_characters(text):
    """
    Remove special characters from a string, keeping only alphanumeric and spaces.
    """
    if not isinstance(text, str):
        return text
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def normalize_column(df, column_name):
    """
    Apply cleaning and special character removal to a column.
    """
    df = clean_dataframe(df, column_name)
    df[column_name] = df[column_name].apply(remove_special_characters)
    return df

if __name__ == "__main__":
    sample_data = {'Name': [' Alice ', 'bob', 'Alice', 'Charlie!', '  david  ']}
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = normalize_column(df, 'Name')
    print("\nCleaned DataFrame:")
    print(cleaned_df)