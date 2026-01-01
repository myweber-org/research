
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list: A list containing elements (must be hashable).
    
    Returns:
        A new list with duplicates removed.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import re

def clean_text(text):
    """
    Clean and normalize the input text by:
    - Removing leading and trailing whitespace.
    - Replacing multiple spaces/newlines/tabs with a single space.
    - Converting the text to lowercase.
    
    Args:
        text (str): The input text to be cleaned.
    
    Returns:
        str: The cleaned and normalized text.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Replace multiple whitespace characters (spaces, newlines, tabs) with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return textimport pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by removing non-numeric characters.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def validate_data(df, required_columns):
    """
    Validate that required columns exist in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if all required columns exist, False otherwise.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True