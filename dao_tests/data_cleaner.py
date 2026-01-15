
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: dictionary to rename columns
        drop_duplicates: whether to remove duplicate rows
        normalize_text: whether to normalize text columns
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def _normalize_string(text):
    """Normalize a string by converting to lowercase and removing extra whitespace."""
    if pd.isna(text):
        return text
    
    normalized = str(text).lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for required columns and minimum row count.
    
    Args:
        df: DataFrame to validate
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(df) < min_rows:
        return False, f"DataFrame has only {len(df)} rows, minimum required is {min_rows}"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def sample_dataframe(df, sample_size=1000, random_state=42):
    """
    Sample a DataFrame if it exceeds a certain size.
    
    Args:
        df: DataFrame to sample
        sample_size: maximum size of sample
        random_state: random seed for reproducibility
    
    Returns:
        Sampled DataFrame
    """
    if len(df) <= sample_size:
        return df
    
    return df.sample(n=sample_size, random_state=random_state)