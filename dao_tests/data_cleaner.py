
import pandas as pd
import re

def clean_dataframe(df, text_column):
    """
    Remove duplicate rows and normalize text in specified column.
    """
    # Remove duplicates
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize text: lowercase and remove extra whitespace
    df_clean[text_column] = df_clean[text_column].apply(
        lambda x: re.sub(r'\s+', ' ', str(x).strip().lower())
    )
    
    return df_clean

def validate_email_column(df, email_column):
    """
    Validate email format in specified column.
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['valid_email'] = df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x)))
    )
    return df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['Alice', 'Bob', 'Alice', 'Charlie'],
        'email': ['alice@example.com', 'bob@test.org', 'alice@example.com', 'invalid-email'],
        'notes': ['  Hello  World  ', 'TEST data', '  hello  world  ', 'Another note']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df, 'notes')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validated_df = validate_email_column(cleaned_df, 'email')
    print("\nDataFrame with email validation:")
    print(validated_df[['name', 'email', 'valid_email']])