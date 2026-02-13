
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing statistical measures.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 14, 13, 12, 11, 10, 9, 8, 12, 13]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nOriginal Statistics:")
    print(calculate_basic_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Statistics:")
    print(calculate_basic_stats(cleaned_df, 'values'))
import re
import pandas as pd
from typing import List, Optional

def remove_special_characters(text: str, keep_chars: str = "") -> str:
    """
    Remove special characters from a string, optionally keeping specified characters.

    Args:
        text: Input string to clean.
        keep_chars: String of characters to preserve (e.g., ".,!?").

    Returns:
        Cleaned string with special characters removed.
    """
    if not isinstance(text, str):
        return text

    pattern = f"[^A-Za-z0-9\\s{re.escape(keep_chars)}]"
    return re.sub(pattern, "", text)

def normalize_whitespace(text: str) -> str:
    """
    Replace multiple whitespace characters with a single space and strip leading/trailing spaces.

    Args:
        text: Input string to normalize.

    Returns:
        String with normalized whitespace.
    """
    if not isinstance(text, str):
        return text

    return " ".join(text.split())

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names: lowercase, replace spaces with underscores, remove special characters.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with cleaned column names.
    """
    df_copy = df.copy()
    df_copy.columns = (
        df_copy.columns.astype(str)
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.strip("_")
    )
    return df_copy

def drop_missing_rows(df: pd.DataFrame, columns: Optional[List[str]] = None, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop rows with missing values based on a threshold.

    Args:
        df: Input DataFrame.
        columns: List of columns to consider. If None, all columns are used.
        threshold: Maximum fraction of missing values allowed per row (0 to 1).

    Returns:
        DataFrame with rows exceeding the missing value threshold removed.
    """
    if columns is None:
        columns = df.columns.tolist()

    missing_ratio = df[columns].isnull().mean(axis=1)
    return df[missing_ratio <= threshold].reset_index(drop=True)

def convert_to_numeric(df: pd.DataFrame, columns: List[str], errors: str = "coerce") -> pd.DataFrame:
    """
    Convert specified columns to numeric type.

    Args:
        df: Input DataFrame.
        columns: List of column names to convert.
        errors: How to handle parsing errors ('coerce', 'raise', 'ignore').

    Returns:
        DataFrame with converted columns.
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors=errors)
    return df_copy

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = "first") -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.

    Args:
        df: Input DataFrame.
        subset: Columns to consider for duplicates. If None, all columns are used.
        keep: Which duplicates to keep ('first', 'last', False).

    Returns:
        DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)