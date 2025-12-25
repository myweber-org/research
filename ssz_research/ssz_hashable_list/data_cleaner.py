import pandas as pd
import re

def clean_dataset(df, text_column):
    """
    Clean a dataset by removing duplicate rows and standardizing text in a specified column.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Standardize text: lowercase, remove extra whitespace
    if text_column in df_cleaned.columns:
        df_cleaned[text_column] = df_cleaned[text_column].apply(
            lambda x: re.sub(r'\s+', ' ', str(x).strip().lower())
        )
    
    return df_cleaned

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column and return a boolean mask.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].str.match(email_pattern, na=False)

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = pd.DataFrame({
#         'name': ['Alice', 'Bob', 'Alice', 'Charlie'],
#         'email': ['alice@example.com', 'bob@test.org', 'alice@example.com', 'invalid-email']
#     })
#     cleaned = clean_dataset(sample_data, 'name')
#     valid_emails = validate_email_column(cleaned, 'email')
#     print(cleaned)
#     print(valid_emails)