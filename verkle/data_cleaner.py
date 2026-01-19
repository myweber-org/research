
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
                        column_type_map: dict) -> pd.DataFrame:
    """
    Convert specified columns to specified data types.
    
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
                         strategy: str = 'mean',
                         columns: List[str] = None) -> pd.DataFrame:
    """
    Handle missing values using specified strategy.
    
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
        if column in df_copy.columns and df_copy[column].isnull().any():
            if strategy == 'mean':
                df_copy[column].fillna(df_copy[column].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[column].fillna(df_copy[column].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[column].fillna(df_copy[column].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[column])
    
    return df_copy

def normalize_numeric_columns(df: pd.DataFrame,
                            columns: List[str] = None,
                            method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize numeric columns using specified method.
    
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
        if column in df_copy.columns and np.issubdtype(df_copy[column].dtype, np.number):
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
                   normalize: bool = False,
                   normalize_method: str = 'minmax') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        type_conversions: Dictionary of column type conversions
        missing_strategy: Strategy for handling missing values
        normalize: Whether to normalize numeric columns
        normalize_method: Normalization method
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize:
        cleaned_df = normalize_numeric_columns(cleaned_df, method=normalize_method)
    
    return cleaned_df
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Load a CSV file, clean missing values, and save the cleaned data.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print("Missing values per column:")
        print(missing_values[missing_values > 0])
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()))
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        print(f"Removed {duplicates_removed} duplicate rows")
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        print(f"Final data shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df):
    """
    Perform basic data validation checks.
    """
    if df is None:
        return False
    
    validation_passed = True
    
    # Check for remaining missing values
    if df.isnull().sum().sum() > 0:
        print("Warning: Data still contains missing values")
        validation_passed = False
    
    # Check for negative values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            print(f"Warning: Column '{col}' contains negative values")
    
    return validation_passed

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_csv, output_csv)
    
    if cleaned_df is not None:
        is_valid = validate_data(cleaned_df)
        if is_valid:
            print("Data cleaning completed successfully")
        else:
            print("Data cleaning completed with warnings")