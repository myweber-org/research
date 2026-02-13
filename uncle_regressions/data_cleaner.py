
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_normalization='lower'):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns_to_clean (list, optional): List of column names to normalize. If None, all object dtype columns are used.
    remove_duplicates (bool): If True, remove duplicate rows.
    case_normalization (str): One of 'lower', 'upper', or None to specify case normalization.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()

    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")

    if columns_to_clean is None:
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns.tolist()

    for col in columns_to_clean:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].astype(str)
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))

            if case_normalization == 'lower':
                cleaned_df[col] = cleaned_df[col].str.lower()
            elif case_normalization == 'upper':
                cleaned_df[col] = cleaned_df[col].str.upper()

            print(f"Cleaned column: {col}")

    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    email_column (str): Name of the column containing email addresses.

    Returns:
    pd.DataFrame: DataFrame with an additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")

    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].astype(str).str.match(email_pattern)
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['Alice', 'Bob', 'Alice', '  charlie  ', 'DAVE'],
        'email': ['alice@example.com', 'invalid-email', 'alice@example.com', 'charlie@test.org', 'DAVE@DOMAIN.COM'],
        'value': [1, 2, 1, 3, 4]
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)

    cleaned = clean_dataframe(df, case_normalization='lower')
    print("\nCleaned DataFrame:")
    print(cleaned)

    validated = validate_email_column(cleaned, 'email')
    print("\nDataFrame with email validation:")
    print(validated)