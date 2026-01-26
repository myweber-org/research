import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase and removing extra whitespace.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def validate_email_column(df, column_name):
    """
    Validate email format in specified column.
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df['is_valid_email'] = df[column_name].str.match(email_pattern)
    return df

def main():
    # Example usage
    data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson  '],
        'email': ['john@example.com', 'jane@example.com', 'invalid-email', 'bob@example.com']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    df = clean_text_column(df, 'name')
    df = remove_duplicates(df, subset=['name'])
    df = validate_email_column(df, 'email')
    
    print("\nCleaned DataFrame:")
    print(df)

if __name__ == "__main__":
    main()