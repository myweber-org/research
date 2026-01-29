
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_mapping (dict, optional): Dictionary mapping original column names to new names.
        drop_duplicates (bool): Whether to remove duplicate rows.
        normalize_text (bool): Whether to normalize text columns (strip, lower case, remove extra spaces).
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x.strip().lower()))
    
    return cleaned_df

def validate_email(email):
    """
    Validate email format using regex.
    
    Args:
        email (str): Email address to validate.
    
    Returns:
        bool: True if email format is valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def filter_valid_emails(df, email_column):
    """
    Filter DataFrame to only include rows with valid email addresses.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with only valid email rows.
    """
    valid_mask = df[email_column].apply(validate_email)
    return df[valid_mask].reset_index(drop=True)

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson  '],
        'email': ['john@example.com', 'jane@example.com', 'invalid-email', 'bob@example.com'],
        'age': [25, 30, 25, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataframe(df, drop_duplicates=True, normalize_text=True)
    print(cleaned)
    print("\nDataFrame with valid emails only:")
    valid_emails = filter_valid_emails(cleaned, 'email')
    print(valid_emails)