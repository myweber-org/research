
import pandas as pd

def clean_dataset(df):
    """
    Remove null values and duplicate rows from a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with no nulls or duplicates.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = df.copy()
    
    cleaned_df = cleaned_df.dropna()
    
    cleaned_df = cleaned_df.drop_duplicates()
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_cleaned_data(df):
    """
    Validate that the DataFrame has no null values or duplicates.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        bool: True if DataFrame is clean, False otherwise.
    """
    has_nulls = df.isnull().any().any()
    has_duplicates = df.duplicated().any()
    
    return not (has_nulls or has_duplicates)

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, None, 4, 2],
        'B': [5, 6, 7, None, 6],
        'C': [8, 9, 10, 11, 9]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nNull values:", df.isnull().sum().sum())
    print("Duplicates:", df.duplicated().sum())
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nValidation result:", validate_cleaned_data(cleaned))