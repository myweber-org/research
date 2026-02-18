
import pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(df: pd.DataFrame, subset: Union[List[str], str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Column(s) to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
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
                print(f"Warning: Could not convert column '{column}' to {dtype}")
    return df_copy

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', fill_value: any = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: Value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(fill_value)
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def clean_numeric_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Clean outliers in numeric column using specified method.
    
    Args:
        df: Input DataFrame
        column: Column name to clean
        method: 'iqr' for interquartile range, 'zscore' for standard deviation
        threshold: Threshold multiplier for outlier detection
    
    Returns:
        DataFrame with outliers handled
    """
    if column not in df.columns:
        return df
    
    numeric_data = pd.to_numeric(df[column], errors='coerce')
    
    if method == 'iqr':
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (numeric_data >= lower_bound) & (numeric_data <= upper_bound)
    elif method == 'zscore':
        mean = numeric_data.mean()
        std = numeric_data.std()
        z_scores = np.abs((numeric_data - mean) / std)
        mask = z_scores <= threshold
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    df_clean = df.copy()
    df_clean[column] = np.where(mask, numeric_data, np.nan)
    return df_clean

def standardize_text_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Standardize text column by converting to lowercase and stripping whitespace.
    
    Args:
        df: Input DataFrame
        column: Column name to standardize
    
    Returns:
        DataFrame with standardized text column
    """
    if column in df.columns:
        df_copy = df.copy()
        df_copy[column] = df_copy[column].astype(str).str.lower().str.strip()
        return df_copy
    return df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame and return summary statistics.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.astype(str).to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        validation_results['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return validation_results