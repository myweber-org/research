
import pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df: pd.DataFrame, 
                         column_type_map: dict) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
    Args:
        df: Input DataFrame
        column_type_map: Dictionary mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_copy = df.copy()
    for column, dtype in column_type_map.items():
        if column in df_copy.columns:
            try:
                df_copy[column] = df_copy[column].astype(dtype)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert column '{column}' to {dtype}")
    return df_copy

def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'drop',
                         fill_value: Union[int, float, str] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: Value to fill missing entries with
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill' and fill_value is not None:
        return df.fillna(fill_value)
    else:
        raise ValueError("Invalid strategy or missing fill_value")

def normalize_column(df: pd.DataFrame, 
                    column: str,
                    method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize a column using specified method.
    
    Args:
        df: Input DataFrame
        column: Column to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        col_min = df_copy[column].min()
        col_max = df_copy[column].max()
        if col_max != col_min:
            df_copy[column] = (df_copy[column] - col_min) / (col_max - col_min)
    
    elif method == 'zscore':
        col_mean = df_copy[column].mean()
        col_std = df_copy[column].std()
        if col_std > 0:
            df_copy[column] = (df_copy[column] - col_mean) / col_std
    
    return df_copy

def clean_dataframe(df: pd.DataFrame,
                   remove_dups: bool = True,
                   type_conversions: dict = None,
                   missing_strategy: str = 'drop',
                   normalize_cols: dict = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        remove_dups: Whether to remove duplicates
        type_conversions: Column type conversion mapping
        missing_strategy: Strategy for handling missing values
        normalize_cols: Dictionary of columns to normalize with methods
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_cols:
        for column, method in normalize_cols.items():
            cleaned_df = normalize_column(cleaned_df, column, method)
    
    return cleaned_df

def validate_dataframe(df: pd.DataFrame,
                      required_columns: List[str] = None,
                      min_rows: int = 1) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        min_rows: Minimum number of rows required
    
    Returns:
        Boolean indicating if DataFrame is valid
    """
    if len(df) < min_rows:
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, 20.3, 20.3, None, 40.7],
        'category': ['A', 'B', 'B', 'C', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    
    cleaned = clean_dataframe(
        df,
        type_conversions={'id': 'int32', 'value': 'float64'},
        missing_strategy='fill',
        normalize_cols={'value': 'minmax'}
    )
    
    print(cleaned)
    print(f"\nData validation: {validate_dataframe(cleaned, ['id', 'value'])}")