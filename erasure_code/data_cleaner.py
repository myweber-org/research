
import pandas as pd
import numpy as np

def clean_dataset(df, text_columns=None, fill_na=True):
    """
    Clean a pandas DataFrame by handling missing values and standardizing text.
    
    Args:
        df: pandas DataFrame to clean
        text_columns: list of column names containing text data
        fill_na: boolean indicating whether to fill missing values
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values
    if fill_na:
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif df_clean[col].dtype == 'object':
                df_clean[col].fillna('Unknown', inplace=True)
    
    # Standardize text columns
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
    
    # Remove duplicate rows
    df_clean.drop_duplicates(inplace=True)
    
    # Reset index after cleaning
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {},
        'data_types': {}
    }
    
    # Check required columns
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    # Count null values
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_results['null_counts'][col] = null_count
    
    # Record data types
    for col in df.columns:
        validation_results['data_types'][col] = str(df[col].dtype)
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'name': ['Alice', 'Bob', None, 'Charlie', 'Alice'],
        'age': [25, 30, np.nan, 35, 25],
        'city': ['New York', 'los angeles', 'Chicago', None, 'New York']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, text_columns=['name', 'city'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned_df, required_columns=['name', 'age', 'city'])
    print("Validation Results:")
    print(validation)