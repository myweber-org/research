import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for nulls. 
                                          If None, checks all columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove rows with null values
    if columns_to_check is None:
        cleaned_df = cleaned_df.dropna()
    else:
        cleaned_df = cleaned_df.dropna(subset=columns_to_check)
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        dict: Validation results with status and messages.
    """
    validation_result = {
        'is_valid': True,
        'messages': []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['messages'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['messages'].append(f'Missing columns: {missing_columns}')
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        validation_result['messages'].append(f'Null values found: {null_counts[null_counts > 0].to_dict()}')
    
    return validation_result

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
        'age': [25, 30, 35, None, 28, 28],
        'score': [85.5, 92.0, 78.5, 88.0, 95.5, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    # Validate the cleaned dataset
    validation = validate_dataset(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    print("Validation Results:")
    print(f"Is Valid: {validation['is_valid']}")
    print(f"Messages: {validation['messages']}")
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
                         column_types: dict) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
    Args:
        df: Input DataFrame
        column_types: Dictionary mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_converted = df.copy()
    for column, dtype in column_types.items():
        if column in df_converted.columns:
            try:
                df_converted[column] = df_converted[column].astype(dtype)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert column '{column}' to {dtype}")
    return df_converted

def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'drop',
                          fill_value: Union[int, float, str] = None) -> pd.DataFrame:
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
    elif strategy == 'fill' and fill_value is not None:
        return df.fillna(fill_value)
    else:
        return df

def normalize_column(df: pd.DataFrame, 
                     column: str,
                     method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize a column using specified method.
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        return df
    
    df_normalized = df.copy()
    
    if method == 'minmax':
        col_min = df_normalized[column].min()
        col_max = df_normalized[column].max()
        if col_max != col_min:
            df_normalized[column] = (df_normalized[column] - col_min) / (col_max - col_min)
    
    elif method == 'zscore':
        col_mean = df_normalized[column].mean()
        col_std = df_normalized[column].std()
        if col_std > 0:
            df_normalized[column] = (df_normalized[column] - col_mean) / col_std
    
    return df_normalized

def clean_dataframe(df: pd.DataFrame,
                    deduplicate: bool = True,
                    type_conversions: dict = None,
                    missing_strategy: str = 'drop',
                    normalize_columns: List[str] = None) -> pd.DataFrame:
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        type_conversions: Dictionary of column type conversions
        missing_strategy: Strategy for handling missing values
        normalize_columns: List of columns to normalize
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_columns:
        for column in normalize_columns:
            cleaned_df = normalize_column(cleaned_df, column)
    
    return cleaned_dfimport pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep ('first', 'last', False).
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def validate_dataframe(df):
    """
    Basic validation of DataFrame structure.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    if df is None:
        return False
    return True

def clean_dataset(file_path, output_path=None):
    """
    Load, clean, and optionally save a dataset.
    
    Args:
        file_path (str): Path to input CSV file.
        output_path (str, optional): Path to save cleaned CSV.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    
    if not validate_dataframe(df):
        print("Error: Invalid DataFrame")
        return None
    
    original_count = len(df)
    cleaned_df = remove_duplicates(df)
    removed_count = original_count - len(cleaned_df)
    
    print(f"Removed {removed_count} duplicate rows")
    print(f"Original: {original_count} rows, Cleaned: {len(cleaned_df)} rows")
    
    if output_path:
        cleaned_df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4, 4, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'David'],
        'value': [10, 20, 20, 30, 40, 40, 50]
    })
    
    print("Original DataFrame:")
    print(sample_data)
    
    cleaned = remove_duplicates(sample_data, subset=['id', 'name'])
    print("\nCleaned DataFrame:")
    print(cleaned)