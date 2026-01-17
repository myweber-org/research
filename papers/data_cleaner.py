import pandas as pd
import numpy as np
from typing import Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[list] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[list] = None) -> pd.DataFrame:
    """
    Fill missing values using specified strategy.
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            
            df_filled[col] = df[col].fillna(fill_value)
        else:
            df_filled[col] = df[col].fillna('Unknown')
    
    return df_filled

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a numeric column to range [0, 1].
    """
    df_normalized = df.copy()
    col_min = df[column].min()
    col_max = df[column].max()
    
    if col_max != col_min:
        df_normalized[column] = (df[column] - col_min) / (col_max - col_min)
    
    return df_normalized

def clean_dataframe(df: pd.DataFrame, 
                   remove_dups: bool = True,
                   fill_na: bool = True,
                   normalize_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Main cleaning pipeline for DataFrame.
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df = normalize_column(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame and return statistics.
    """
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return stats