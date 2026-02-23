
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_na=None):
    """
    Clean a pandas DataFrame by removing null values and optionally dropping duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_na (str or None): Method to fill null values ('mean', 'median', 'mode', or None for drop).
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle null values
    if fill_na is None:
        cleaned_df = cleaned_df.dropna()
    elif fill_na == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_na == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fill_na == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"