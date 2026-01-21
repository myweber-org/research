import pandas as pd

def clean_dataset(df, drop_duplicates=True):
    """
    Clean a pandas DataFrame by removing null values and optionally duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    cleaned_df = cleaned_df.dropna()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate that DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'name': ['Alice', 'Bob', None, 'Alice', 'Charlie'],
        'age': [25, 30, 35, 25, 40],
        'score': [85.5, 90.0, None, 85.5, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nShape:", df.shape)
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nShape:", cleaned.shape)
    
    is_valid = validate_data(cleaned, ['name', 'age'])
    print(f"\nData validation passed: {is_valid}")