import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing rows with null values
    and dropping duplicate rows.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_check (list, optional): Specific columns to check for nulls/duplicates
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # Remove rows with null values
    if columns_to_check:
        cleaned_df = cleaned_df.dropna(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.dropna()
    
    # Remove duplicate rows
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list, optional): List of required column names
    
    Returns:
    tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"