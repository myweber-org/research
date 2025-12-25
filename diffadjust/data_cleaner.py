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