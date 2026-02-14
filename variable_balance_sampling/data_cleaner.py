
import pandas as pd

def clean_dataset(df, column_names=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_names (list, optional): List of column names to apply string normalization.
            If None, all object dtype columns are normalized.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = cleaned_df.shape[0]
    cleaned_df = cleaned_df.drop_duplicates()
    removed_duplicates = initial_rows - cleaned_df.shape[0]
    
    # Normalize string columns
    if column_names is None:
        # Select all object dtype columns (typically strings)
        string_columns = cleaned_df.select_dtypes(include=['object']).columns
    else:
        # Use provided column names
        string_columns = [col for col in column_names if col in cleaned_df.columns]
    
    for col in string_columns:
        if cleaned_df[col].dtype == 'object':
            # Strip whitespace and convert to lowercase
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    # Print cleaning summary
    print(f"Removed {removed_duplicates} duplicate rows")
    print(f"Normalized {len(string_columns)} string columns")
    print(f"Final dataset shape: {cleaned_df.shape}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    # Check for excessive missing values
    missing_percentage = df.isnull().sum() / len(df) * 100
    high_missing = missing_percentage[missing_percentage > 50]
    
    if not high_missing.empty:
        print(f"Warning: Columns with >50% missing values: {list(high_missing.index)}")
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', '  BOB JOHNSON  ', 'Alice'],
        'age': [25, 30, 25, 35, None],
        'email': ['JOHN@EXAMPLE.COM', 'jane@example.com', 'john@example.com', 'bob@example.com', 'alice@example.com'],
        'city': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Boston']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, column_names=['name', 'email', 'city'])
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    validation_result = validate_dataframe(cleaned, required_columns=['name', 'age'])
    print(f"\nData validation passed: {validation_result}")