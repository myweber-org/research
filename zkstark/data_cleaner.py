
import pandas as pd
import re

def clean_text_column(series, case='lower', remove_special=True):
    """
    Standardize text data in a pandas Series.
    """
    if case == 'lower':
        series = series.str.lower()
    elif case == 'upper':
        series = series.str.upper()
    
    if remove_special:
        series = series.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)))
    
    return series.str.strip()

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def validate_email(series):
    """
    Validate email addresses in a Series.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return series.str.match(pattern)

def main():
    # Example usage
    data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson  '],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.net'],
        'value': [1, 2, 1, 3]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    # Clean text columns
    df['name'] = clean_text_column(df['name'])
    df['email'] = clean_text_column(df['email'], remove_special=False)
    
    # Remove duplicates
    df = remove_duplicates(df, subset=['name'], keep='first')
    
    # Validate emails
    df['valid_email'] = validate_email(df['email'])
    
    print("\nCleaned DataFrame:")
    print(df)

if __name__ == "__main__":
    main()