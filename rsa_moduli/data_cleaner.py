
import pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df: pd.DataFrame, 
                         column_types: dict) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    """
    df_converted = df.copy()
    for column, dtype in column_types.items():
        if column in df_converted.columns:
            try:
                df_converted[column] = df_converted[column].astype(dtype)
            except (ValueError, TypeError):
                df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
    return df_converted

def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'drop',
                          fill_value: Union[int, float, str] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    """
    df_clean = df.copy()
    
    if strategy == 'drop':
        df_clean = df_clean.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            df_clean = df_clean.fillna(fill_value)
        else:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    return df_clean

def clean_dataframe(df: pd.DataFrame,
                    deduplicate: bool = True,
                    type_conversions: dict = None,
                    missing_strategy: str = 'drop') -> pd.DataFrame:
    """
    Main function to clean DataFrame with multiple operations.
    """
    df_clean = df.copy()
    
    if deduplicate:
        df_clean = remove_duplicates(df_clean)
    
    if type_conversions:
        df_clean = convert_column_types(df_clean, type_conversions)
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy)
    
    return df_clean

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame structure and content.
    """
    if df.empty:
        return False
    
    if df.isnull().all().any():
        return False
    
    return True