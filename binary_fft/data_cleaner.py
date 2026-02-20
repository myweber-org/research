
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_text_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize text column by converting to lowercase and stripping whitespace.
    
    Args:
        df: Input DataFrame
        column: Name of column to normalize
    
    Returns:
        DataFrame with normalized column
    """
    df = df.copy()
    df[column] = df[column].astype(str).str.lower().str.strip()
    return df

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in DataFrame columns.
    
    Args:
        df: Input DataFrame
        strategy: Method to use for filling ('mean', 'median', 'mode', 'constant')
        columns: Specific columns to fill, fills all numeric columns if None
    
    Returns:
        DataFrame with missing values filled
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            if strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == 'constant':
                df[col].fillna(0, inplace=True)
    
    return df

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        True if all required columns are present, False otherwise
    """
    return all(col in df.columns for col in required_columns)

def clean_dataset(df: pd.DataFrame, 
                  text_columns: Optional[List[str]] = None,
                  numeric_strategy: str = 'mean') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        text_columns: Columns containing text data to normalize
        numeric_strategy: Strategy for filling missing numeric values
    
    Returns:
        Cleaned DataFrame
    """
    # Remove duplicates
    df_clean = remove_duplicates(df)
    
    # Normalize text columns
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean = normalize_text_column(df_clean, col)
    
    # Fill missing values
    df_clean = fill_missing_values(df_clean, strategy=numeric_strategy)
    
    return df_clean

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'BOB', 'alice', 'Charlie ', 'David', 'David'],
        'age': [25, 30, None, 35, 40, 40],
        'score': [85.5, 92.0, None, 78.5, 88.0, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df, text_columns=['name'], numeric_strategy='mean')
    print(cleaned_df)