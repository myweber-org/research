import pandas as pd
import numpy as np

def clean_dataset(df, text_columns=None):
    """
    Clean a pandas DataFrame by removing rows with null values
    and standardizing text columns to lowercase.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove rows with any null values
    cleaned_df = cleaned_df.dropna()
    
    # Standardize text columns if specified
    if text_columns:
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].astype(str).str.lower().str.strip()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_cleaned_data(df):
    """
    Validate that the cleaned DataFrame has no null values
    and text columns are properly standardized.
    """
    # Check for null values
    if df.isnull().sum().sum() > 0:
        return False, "DataFrame contains null values"
    
    # Check that all values are strings in text columns
    # (assuming we want to validate string columns)
    for col in df.select_dtypes(include=['object']).columns:
        if not all(isinstance(val, str) for val in df[col]):
            return False, f"Column {col} contains non-string values"
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'name': ['John', 'Jane', None, 'Bob', 'Alice'],
        'age': [25, 30, 35, None, 28],
        'city': ['New York', 'LONDON', 'Paris', 'Berlin', 'TOKYO']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, text_columns=['name', 'city'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid, message = validate_cleaned_data(cleaned_df)
    print(f"\nValidation: {message}")