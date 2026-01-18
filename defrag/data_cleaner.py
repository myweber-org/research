
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicate_rows(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame with configurable parameters.
    
    Args:
        df: Input pandas DataFrame
        subset: Column labels to consider for identifying duplicates
        keep: Which duplicates to keep - 'first', 'last', or False
        inplace: Whether to modify the DataFrame in place
    
    Returns:
        DataFrame with duplicates removed
    """
    if not inplace:
        df = df.copy()
    
    if subset is None:
        subset = df.columns.tolist()
    
    # Remove duplicates
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    if inplace:
        df.drop(df.index, inplace=True)
        df.update(cleaned_df)
        return df
    
    return cleaned_df

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Basic validation of DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Boolean indicating if DataFrame is valid
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    # Check for infinite values
    if np.any(np.isinf(df.select_dtypes(include=[np.number]))):
        print("Warning: DataFrame contains infinite values")
    
    return True

def clean_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Clean numeric columns by converting to appropriate dtype and handling NaN.
    
    Args:
        df: Input DataFrame
        columns: List of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            # Convert to numeric, coercing errors to NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            # Fill NaN with column mean
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    return df_clean

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'Eve'],
        'score': [85, 90, 90, 78, 92, 92, 88],
        'age': [25, 30, 30, 22, 35, 35, 28]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nShape:", df.shape)
    
    # Remove duplicates based on 'id' and 'name'
    cleaned_df = remove_duplicate_rows(df, subset=['id', 'name'], keep='first')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nShape after cleaning:", cleaned_df.shape)
    
    # Validate the cleaned DataFrame
    is_valid = validate_dataframe(cleaned_df)
    print(f"\nDataFrame validation: {is_valid}")