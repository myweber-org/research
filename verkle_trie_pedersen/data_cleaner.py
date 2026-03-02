
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (bool): Whether to fill missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
            elif cleaned_df[column].dtype == 'object':
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
        print("Missing values have been filled")
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate the dataset for required columns and data integrity.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.isnull().sum().sum() > 0:
        print("Dataset contains missing values")
        return False
    
    return True

def main():
    sample_data = {
        'id': [1, 2, 3, 4, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', None, 'David', 'Eve'],
        'age': [25, 30, None, 35, 35, 28],
        'score': [85.5, 92.0, 78.5, 88.0, 88.0, 91.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    is_valid = validate_dataset(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    print(f"\nDataset validation: {'PASS' if is_valid else 'FAIL'}")

if __name__ == "__main__":
    main()