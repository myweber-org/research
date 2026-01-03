
import pandas as pd
import re

def clean_dataframe(df, text_column):
    """
    Remove duplicate rows and normalize text in specified column.
    """
    # Remove duplicates
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize text: lowercase, remove extra whitespace
    df_clean[text_column] = df_clean[text_column].apply(
        lambda x: re.sub(r'\s+', ' ', str(x).strip().lower())
    )
    
    return df_clean

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4],
        'text': ['Hello World', 'Python Code', 'python code', '  DATA  ', 'Test']
    })
    
    cleaned = clean_dataframe(sample_data, 'text')
    save_cleaned_data(cleaned, 'cleaned_data.csv')
    
    print("Original shape:", sample_data.shape)
    print("Cleaned shape:", cleaned.shape)
    print("\nCleaned data preview:")
    print(cleaned.head())import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase, removing extra whitespace,
    and stripping special characters except basic punctuation.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s.,!?-]', '', x))
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    return df

def remove_duplicate_rows(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def validate_email_column(df, column_name):
    """
    Validate email format and return boolean mask.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[column_name].str.match(email_pattern)

def main():
    # Example usage
    data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson  '],
        'email': ['john@example.com', 'invalid-email', 'john@example.com', 'bob@company.org'],
        'notes': ['Important client!!!', 'Needs follow-up.', 'Important client!!!', '  Regular customer  ']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean text columns
    df = clean_text_column(df, 'name')
    df = clean_text_column(df, 'notes')
    
    # Remove duplicates
    df = remove_duplicate_rows(df, subset=['name', 'email'])
    
    # Validate emails
    df['valid_email'] = validate_email_column(df, 'email')
    
    print("Cleaned DataFrame:")
    print(df)

if __name__ == "__main__":
    main()