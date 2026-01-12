import pandas as pd

def clean_dataset(df, columns_to_check=None, remove_duplicates=True):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for nulls.
            If None, checks all columns. Defaults to None.
        remove_duplicates (bool, optional): Whether to remove duplicate rows.
            Defaults to True.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove null values
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns
    
    cleaned_df = cleaned_df.dropna(subset=columns_to_check)
    
    # Remove duplicates
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'name': ['Alice', 'Bob', None, 'Alice', 'Charlie'],
#         'age': [25, 30, 35, 25, 40],
#         'city': ['NYC', 'LA', 'Chicago', 'NYC', 'Boston']
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame:")
#     cleaned = clean_dataset(df)
#     print(cleaned)
#     
#     # Validate
#     is_valid, message = validate_dataframe(cleaned, ['name', 'age'])
#     print(f"\nValidation: {is_valid}, Message: {message}")