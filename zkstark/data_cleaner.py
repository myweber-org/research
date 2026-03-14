
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
import re
import pandas as pd

def clean_text_column(df, column_name):
    """
    Standardize text in a DataFrame column: lowercase, strip whitespace, remove extra spaces.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x))
    return df

def remove_duplicate_rows(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

def example_usage():
    # Example data
    data = {
        'name': ['  Alice  ', 'Bob', 'alice', '  CAROL  ', 'Bob   '],
        'age': [25, 30, 25, 35, 30],
        'city': ['New York', 'London', 'new york', 'Paris', 'london']
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    # Clean text in 'name' and 'city' columns
    df = clean_text_column(df, 'name')
    df = clean_text_column(df, 'city')
    
    # Remove duplicates based on 'name' and 'age'
    df = remove_duplicate_rows(df, subset=['name', 'age'])
    
    print("\nCleaned DataFrame:")
    print(df)
    
    # Save to file (optional)
    # save_cleaned_data(df, 'cleaned_data.csv')

if __name__ == "__main__":
    example_usage()