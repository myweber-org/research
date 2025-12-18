
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(
    data: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    data: Input DataFrame
    subset: Column labels to consider for identifying duplicates
    keep: Which duplicates to keep - 'first', 'last', or False
    inplace: Whether to modify the DataFrame in place
    
    Returns:
    DataFrame with duplicates removed
    """
    if not inplace:
        data = data.copy()
    
    if subset is None:
        subset = data.columns.tolist()
    
    cleaned_data = data.drop_duplicates(subset=subset, keep=keep)
    
    if inplace:
        data.drop(data.index, inplace=True)
        data.update(cleaned_data)
        return data
    
    return cleaned_data

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df: DataFrame to validate
    
    Returns:
    Boolean indicating if DataFrame is valid
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return True
    
    if df.isnull().all().all():
        return False
    
    return True

def clean_numeric_columns(
    df: pd.DataFrame,
    columns: List[str],
    fill_method: str = 'mean'
) -> pd.DataFrame:
    """
    Clean numeric columns by handling missing values.
    
    Parameters:
    df: Input DataFrame
    columns: List of numeric column names to clean
    fill_method: Method to fill missing values - 'mean', 'median', or 'zero'
    
    Returns:
    DataFrame with cleaned numeric columns
    """
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
        
        if fill_method == 'mean':
            fill_value = df_clean[col].mean()
        elif fill_method == 'median':
            fill_value = df_clean[col].median()
        elif fill_method == 'zero':
            fill_value = 0
        else:
            fill_value = df_clean[col].mean()
        
        df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David'],
        'score': [85, 90, 90, 78, None, 92],
        'age': [25, 30, 30, 22, 35, 35]
    })
    
    print("Original data:")
    print(sample_data)
    print("\nAfter removing duplicates:")
    cleaned = remove_duplicates(sample_data, subset=['id', 'name'])
    print(cleaned)
    
    print("\nAfter cleaning numeric columns:")
    numeric_cleaned = clean_numeric_columns(cleaned, columns=['score'], fill_method='mean')
    print(numeric_cleaned)