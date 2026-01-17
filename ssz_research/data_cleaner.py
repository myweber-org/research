
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    For numeric columns, missing values are filled with the column median.
    For categorical columns, missing values are filled with the most frequent value.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Separate numeric and categorical columns
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    categorical_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
    
    # Fill missing values for numeric columns with median
    for col in numeric_cols:
        if df_cleaned[col].isnull().any():
            median_value = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_value)
    
    # Fill missing values for categorical columns with mode
    for col in categorical_cols:
        if df_cleaned[col].isnull().any():
            mode_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
            df_cleaned[col] = df_cleaned[col].fillna(mode_value)
    
    return df_cleaned

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame structure and content.
    Checks for required columns and ensures no missing values remain.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for any remaining missing values
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0].index.tolist()
        raise ValueError(f"Data still contains missing values in columns: {null_cols}")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, None, 35, 40],
        'score': [85.5, 92.0, 92.0, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nMissing values count:")
    print(df.isnull().sum())
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_data(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
        print("\nData validation passed!")
    except ValueError as e:
        print(f"\nData validation failed: {e}")