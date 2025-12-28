
import pandas as pd

def clean_dataset(df, remove_nulls=True, remove_duplicates=True):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        remove_nulls (bool): If True, drop rows with any null values.
        remove_duplicates (bool): If True, drop duplicate rows.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if remove_nulls:
        cleaned_df = cleaned_df.dropna()
    
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate that DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

def normalize_column_names(df):
    """
    Normalize column names to lowercase with underscores.
    
    Args:
        df (pd.DataFrame): DataFrame with columns to normalize.
    
    Returns:
        pd.DataFrame: DataFrame with normalized column names.
    """
    df = df.copy()
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df