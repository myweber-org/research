
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe: Input pandas DataFrame
        subset: Column label or sequence of labels to consider for duplicates
        keep: Determines which duplicates to mark
            'first' : Mark duplicates as True except for the first occurrence
            'last' : Mark duplicates as True except for the last occurrence
            False : Mark all duplicates as True
    
    Returns:
        Cleaned DataFrame with duplicates removed
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by removing non-numeric values and converting to float.
    
    Args:
        dataframe: Input pandas DataFrame
        columns: List of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    cleaned_df = dataframe.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: Input pandas DataFrame
        required_columns: List of required column names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if dataframe.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"