import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_method='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to clean.
    columns_to_check (list, optional): List of column names to check for duplicates.
                                       If None, checks all columns.
    fill_method (str): Method to fill missing values. Options: 'mean', 'median', 'mode', 'zero'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if columns_to_check is None:
        cleaned_df = cleaned_df.drop_duplicates()
    else:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    
    # Handle missing values
    for column in cleaned_df.columns:
        if cleaned_df[column].dtype in ['int64', 'float64']:
            if cleaned_df[column].isnull().any():
                if fill_method == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_method == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_method == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                elif fill_method == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unsupported fill method: {fill_method}")
                
                cleaned_df[column].fillna(fill_value, inplace=True)
        else:
            # For non-numeric columns, fill with the most frequent value
            if cleaned_df[column].isnull().any():
                mode_value = cleaned_df[column].mode()
                if not mode_value.empty:
                    cleaned_df[column].fillna(mode_value[0], inplace=True)
                else:
                    cleaned_df[column].fillna('Unknown', inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to validate.
    required_columns (list, optional): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True  # Empty DataFrame is valid
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4, 5],
#         'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
#         'age': [25, 30, 30, None, 35, 40],
#         'score': [85.5, 92.0, 92.0, 78.5, None, 95.0]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame:")
#     cleaned = clean_dataset(df, fill_method='mean')
#     print(cleaned)
def remove_duplicates(input_list):
    """
    Remove duplicate items from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def clean_numeric_strings(string_list):
    """
    Clean a list of numeric strings by converting to integers,
    removing non-numeric entries, and returning sorted unique values.
    """
    cleaned = []
    for item in string_list:
        try:
            num = int(item.strip())
            cleaned.append(num)
        except (ValueError, AttributeError):
            continue
    
    unique_nums = remove_duplicates(cleaned)
    return sorted(unique_nums)

def validate_email_list(email_list):
    """
    Basic email validation and cleaning.
    Returns a list of valid email addresses (simple validation).
    """
    valid_emails = []
    for email in email_list:
        if not isinstance(email, str):
            continue
        email = email.strip().lower()
        if '@' in email and '.' in email.split('@')[-1]:
            valid_emails.append(email)
    return remove_duplicates(valid_emails)