
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for duplicates.
            If None, checks all columns. Defaults to None.
        fill_missing (str or value): Method to fill missing values.
            Options: 'mean', 'median', 'mode', or a specific value. Defaults to 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    original_shape = df.shape
    
    # Remove duplicates
    df_cleaned = df.drop_duplicates(subset=columns_to_check, keep='first')
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    
    # Handle missing values
    missing_before = df_cleaned.isnull().sum().sum()
    
    if fill_missing == 'mean':
        df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
    elif fill_missing == 'median':
        df_cleaned = df_cleaned.fillna(df_cleaned.median(numeric_only=True))
    elif fill_missing == 'mode':
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                mode_val = df_cleaned[col].mode()
                if not mode_val.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
    else:
        df_cleaned = df_cleaned.fillna(fill_missing)
    
    missing_after = df_cleaned.isnull().sum().sum()
    
    # Print cleaning summary
    print(f"Original dataset shape: {original_shape}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Missing values before: {missing_before}")
    print(f"Missing values after: {missing_after}")
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Error: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    data = {
        'id': [1, 2, 3, 1, 2, 6],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'David'],
        'age': [25, 30, None, 25, 30, 35],
        'score': [85.5, 92.0, 78.5, 85.5, 92.0, None]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'], fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    print(f"\nData validation passed: {is_valid}")