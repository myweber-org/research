
import pandas as pd
import re

def clean_dataset(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping original column names to new names
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip, lower case)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
            cleaned_df[col] = cleaned_df[col].replace(r'\s+', ' ', regex=True)
            print(f"Normalized text in column: {col}")
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        email_column (str): Name of the column containing email addresses
    
    Returns:
        pd.DataFrame: DataFrame with validation results
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validation_df = df.copy()
    validation_df['email_valid'] = validation_df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    valid_count = validation_df['email_valid'].sum()
    total_count = len(validation_df)
    
    print(f"Email validation results:")
    print(f"  Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    print(f"  Invalid emails: {total_count - valid_count}/{total_count} ({(total_count - valid_count)/total_count*100:.1f}%)")
    
    return validation_df

def sample_data_cleaning():
    """
    Example usage of the data cleaning functions.
    """
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', 'ALICE WONDER'],
        'email': ['john@example.com', 'jane@example.com', 'john@example.com', 'bob@example', 'alice@example.com'],
        'age': [25, 30, 25, 35, 28],
        'city': ['New York', 'Los Angeles', 'New York', 'Chicago', 'BOSTON']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, normalize_text=True)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validated = validate_email_column(cleaned, 'email')
    print("\nDataFrame with email validation:")
    print(validated[['name', 'email', 'email_valid']])
    
    return cleaned, validated

if __name__ == "__main__":
    cleaned_df, validated_df = sample_data_cleaning()
def filter_valid_entries(data_list, required_keys):
    """
    Returns a new list containing only dictionaries that have all specified keys
    and where none of the required key values are None or empty strings.
    """
    if not isinstance(data_list, list):
        raise TypeError("Input must be a list")
    if not isinstance(required_keys, list):
        raise TypeError("Required keys must be a list")

    filtered_data = []
    for entry in data_list:
        if not isinstance(entry, dict):
            continue

        is_valid = True
        for key in required_keys:
            if key not in entry:
                is_valid = False
                break
            value = entry.get(key)
            if value is None or (isinstance(value, str) and value.strip() == ""):
                is_valid = False
                break

        if is_valid:
            filtered_data.append(entry)

    return filtered_data