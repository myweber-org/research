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

def convert_column_types(
    df: pd.DataFrame, 
    column_type_map: dict
) -> pd.DataFrame:
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

def handle_missing_values(
    df: pd.DataFrame, 
    strategy: str = 'drop', 
    fill_value: Union[int, float, str] = None
) -> pd.DataFrame:
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
        if fill_value is not None:
            return df.fillna(fill_value)
        else:
            return df.fillna(df.mean(numeric_only=True))
    return df

def clean_dataframe(
    df: pd.DataFrame,
    deduplicate: bool = True,
    type_conversions: dict = None,
    missing_strategy: str = 'drop'
) -> pd.DataFrame:
    """
    Main function to clean DataFrame with multiple operations.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        type_conversions: Dictionary for column type conversions
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None],
        'age': [25, 30, 30, None, 35],
        'score': ['85', '90', '90', '95', '100']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    type_map = {'score': 'int32'}
    cleaned = clean_dataframe(
        df, 
        deduplicate=True,
        type_conversions=type_map,
        missing_strategy='fill'
    )
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nCleaned DataFrame info:")
    print(cleaned.info())
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    if fill_missing:
        for column in df.columns:
            if df[column].isnull().any():
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].mean())
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].median())
                elif fill_missing == 'mode':
                    df[column] = df[column].fillna(df[column].mode()[0])
                elif fill_missing == 'zero':
                    df[column] = df[column].fillna(0)
                else:
                    df[column] = df[column].fillna(method='ffill')
    
    print(f"Dataset cleaned: {original_shape} -> {df.shape}")
    return df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    """
    if df.empty:
        raise ValueError("Dataset is empty")
    
    if len(df) < min_rows:
        raise ValueError(f"Dataset has fewer than {min_rows} rows")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def main():
    # Example usage
    data = {
        'A': [1, 2, 2, 3, np.nan],
        'B': [4, 5, 5, np.nan, 7],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    try:
        validate_dataset(cleaned_df, required_columns=['A', 'B', 'C'], min_rows=1)
        print("Dataset validation passed")
    except ValueError as e:
        print(f"Dataset validation failed: {e}")

if __name__ == "__main__":
    main()