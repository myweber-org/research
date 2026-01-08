
import pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df: pd.DataFrame, 
                         column_types: dict) -> pd.DataFrame:
    """
    Convert specified columns to given data types.
    
    Args:
        df: Input DataFrame
        column_types: Dictionary mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_copy = df.copy()
    for column, dtype in column_types.items():
        if column in df_copy.columns:
            try:
                df_copy[column] = df_copy[column].astype(dtype)
            except (ValueError, TypeError):
                df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
    return df_copy

def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'mean',
                          columns: List[str] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: Specific columns to process
    
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for column in columns:
        if column in df_copy.columns:
            if strategy == 'drop':
                df_copy = df_copy.dropna(subset=[column])
            elif strategy == 'mean':
                df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
            elif strategy == 'median':
                df_copy[column] = df_copy[column].fillna(df_copy[column].median())
            elif strategy == 'mode':
                df_copy[column] = df_copy[column].fillna(df_copy[column].mode()[0])
    
    return df_copy

def normalize_numeric_columns(df: pd.DataFrame,
                             columns: List[str] = None,
                             method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize numeric columns in DataFrame.
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized columns
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns
    
    for column in columns:
        if column in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[column]):
            if method == 'minmax':
                min_val = df_copy[column].min()
                max_val = df_copy[column].max()
                if max_val > min_val:
                    df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df_copy[column].mean()
                std_val = df_copy[column].std()
                if std_val > 0:
                    df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

def clean_dataframe(df: pd.DataFrame,
                   deduplicate: bool = True,
                   type_conversions: dict = None,
                   missing_strategy: str = 'mean',
                   normalize: bool = False) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        type_conversions: Dictionary of column type conversions
        missing_strategy: Strategy for handling missing values
        normalize: Whether to normalize numeric columns
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    if missing_strategy != 'none':
        cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize:
        cleaned_df = normalize_numeric_columns(cleaned_df)
    
    return cleaned_df