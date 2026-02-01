import pandas as pd
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
        strategy: Method for filling missing values ('mean', 'median', 'mode', 'constant')
        columns: Specific columns to fill, fills all numeric columns if None
    
    Returns:
        DataFrame with missing values filled
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        elif strategy == 'constant':
            df[col] = df[col].fillna(0)
    
    return df

def clean_dataset(df: pd.DataFrame, 
                  text_columns: Optional[List[str]] = None,
                  numeric_strategy: str = 'mean') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        text_columns: List of text columns to normalize
        numeric_strategy: Strategy for filling numeric missing values
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    df_clean = remove_duplicates(df_clean)
    
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean = normalize_text_column(df_clean, col)
    
    df_clean = fill_missing_values(df_clean, strategy=numeric_strategy)
    
    return df_clean

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Basic validation of DataFrame structure.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if DataFrame passes validation checks
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    if df.columns.duplicated().any():
        return False
    
    return True