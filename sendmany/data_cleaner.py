
import pandas as pd
import numpy as np
from typing import List, Optional

def clean_dataframe(df: pd.DataFrame, 
                    drop_duplicates: bool = True,
                    fill_missing: Optional[str] = None,
                    columns_to_standardize: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean a pandas DataFrame by removing duplicates, handling missing values,
    and standardizing specified columns.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if fill_missing is not None:
        if fill_missing == 'mean':
            df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
        elif fill_missing == 'median':
            df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        elif fill_missing == 'mode':
            df_clean = df_clean.fillna(df_clean.mode().iloc[0])
        elif fill_missing == 'zero':
            df_clean = df_clean.fillna(0)
    
    if columns_to_standardize:
        for col in columns_to_standardize:
            if col in df_clean.columns and df_clean[col].dtype in ['float64', 'int64']:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                if std > 0:
                    df_clean[col] = (df_clean[col] - mean) / std
    
    return df_clean

def validate_dataframe(df: pd.DataFrame, 
                       required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame contains all required columns and has no NaN values
    in those columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    nan_counts = df[required_columns].isna().sum()
    if nan_counts.any():
        print(f"NaN values found in required columns:\n{nan_counts[nan_counts > 0]}")
        return False
    
    return True

def sample_dataframe(df: pd.DataFrame, 
                     sample_size: int = 1000,
                     random_state: int = 42) -> pd.DataFrame:
    """
    Create a random sample from a DataFrame while maintaining relative distributions.
    """
    if len(df) <= sample_size:
        return df
    
    return df.sample(n=sample_size, random_state=random_state)

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10.5, 20.3, np.nan, 30.1, 40.7, 50.2],
        'category': ['A', 'B', 'A', 'A', 'B', 'C']
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned_data = clean_dataframe(
        sample_data,
        drop_duplicates=True,
        fill_missing='mean',
        columns_to_standardize=['value']
    )
    
    print("\nCleaned data:")
    print(cleaned_data)
    
    is_valid = validate_dataframe(cleaned_data, ['id', 'value'])
    print(f"\nData validation: {is_valid}")
    
    sampled_data = sample_dataframe(cleaned_data, sample_size=3)
    print("\nSampled data:")
    print(sampled_data)
import pandas as pd

def clean_data(df, drop_duplicates=True, fill_missing=True):
    """
    Clean the input DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (bool): Whether to fill missing values. Default is True.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed_rows = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed_rows} duplicate rows.")
    
    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna('Unknown')
        
        print("Filled missing values in numeric and categorical columns.")
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that the DataFrame contains all required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if all required columns are present, False otherwise.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    else:
        print("All required columns are present.")
        return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 3, 4, None],
        'name': ['Alice', 'Bob', 'Charlie', 'Charlie', None, 'Eve'],
        'age': [25, 30, None, 35, 40, 45],
        'score': [85.5, 92.0, 78.5, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_data(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    required_cols = ['id', 'name', 'age']
    is_valid = validate_data(cleaned_df, required_cols)
    print(f"\nData validation result: {is_valid}")