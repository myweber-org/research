
import pandas as pd

def clean_dataset(df, column_mapping=None, drop_duplicates=True):
    """
    Clean a pandas DataFrame by standardizing column names and removing duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_mapping (dict, optional): Dictionary mapping original column names 
                                         to standardized names. Defaults to None.
        drop_duplicates (bool, optional): Whether to remove duplicate rows. 
                                          Defaults to True.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Standardize column names
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Convert column names to lowercase and replace spaces with underscores
    cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(' ', '_')
    
    # Remove duplicate rows
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    # Remove rows with all NaN values
    cleaned_df = cleaned_df.dropna(how='all')
    
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
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Customer ID': [1, 2, 2, 3, 4],
        'First Name': ['John', 'Jane', 'Jane', 'Bob', None],
        'Last Name': ['Doe', 'Smith', 'Smith', 'Johnson', 'Williams'],
        'Email': ['john@example.com', 'jane@example.com', 
                  'jane@example.com', 'bob@example.com', 'will@example.com']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    column_map = {'Customer ID': 'customer_id', 
                  'First Name': 'first_name',
                  'Last Name': 'last_name',
                  'Email': 'email'}
    
    cleaned_df = clean_dataset(df, column_mapping=column_map)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned dataset
    is_valid, message = validate_dataframe(cleaned_df, 
                                          required_columns=['customer_id', 'email'])
    print(f"\nValidation: {is_valid} - {message}")