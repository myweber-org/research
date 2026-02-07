
import pandas as pd
import re

def clean_dataframe(df, column_names=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # If specific columns are provided, clean only those; otherwise, clean all object columns
    if column_names is None:
        column_names = df_clean.select_dtypes(include=['object']).columns
    
    for col in column_names:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(_normalize_string)
    
    return df_clean

def _normalize_string(s):
    """
    Normalize a string: lowercase, strip whitespace, and remove extra spaces.
    """
    if pd.isna(s):
        return s
    s = str(s)
    s = s.lower()
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    Returns a Series with boolean values.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].str.match(email_pattern, na=False)