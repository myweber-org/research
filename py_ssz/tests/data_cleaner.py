import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text in a DataFrame column by converting to lowercase,
    removing extra whitespace, and stripping special characters.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.lower()
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame with optional column subset.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def validate_email_column(df, column_name):
    """
    Validate email addresses in a DataFrame column and return boolean mask.
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[column_name].str.match(email_pattern)

def process_dataframe(df, text_columns=None, email_columns=None):
    """
    Main function to clean and process DataFrame with multiple operations.
    """
    if text_columns:
        for col in text_columns:
            df = clean_text_column(df, col)
    
    df = remove_duplicates(df)
    
    if email_columns:
        for col in email_columns:
            if col in df.columns:
                df[f'{col}_valid'] = validate_email_column(df, col)
    
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith  ', 'ALICE BROWN', 'John Doe'],
        'email': ['john@example.com', 'invalid-email', 'alice@test.org', 'john@example.com'],
        'notes': ['Important client!', 'Needs follow-up.', '  Regular customer  ', 'Important client!']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    processed_df = process_dataframe(
        df, 
        text_columns=['name', 'notes'],
        email_columns=['email']
    )
    
    print("Processed DataFrame:")
    print(processed_df)