
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns_to_clean (list, optional): List of column names to apply text normalization.
                                       If None, all object dtype columns are cleaned.
    remove_duplicates (bool): If True, remove duplicate rows.
    normalize_text (bool): If True, normalize text in specified columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if remove_duplicates:
        initial_rows = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - df_clean.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if normalize_text:
        if columns_to_clean is None:
            columns_to_clean = df_clean.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns_to_clean:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(_normalize_string)
                print(f"Normalized text in column: {col}")
    
    return df_clean

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Parameters:
    text (str): Input string.
    
    Returns:
    str: Normalized string.
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    email_column (str): Name of the column containing email addresses.
    
    Returns:
    pd.DataFrame: DataFrame with validation results.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    df_result = df.copy()
    df_result['email_valid'] = df_result[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    
    valid_count = df_result['email_valid'].sum()
    total_count = df_result.shape[0]
    
    print(f"Email validation: {valid_count} valid out of {total_count} total.")
    
    return df_result