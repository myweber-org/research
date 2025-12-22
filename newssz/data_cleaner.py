
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values - 'mean', 'median', or 'mode'
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if cleaned_df.isnull().sum().any():
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        if fill_missing == 'mean':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        elif fill_missing == 'median':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        elif fill_missing == 'mode':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
        
        print(f"Filled missing values using {fill_missing} method")
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 4],
        'B': [5, None, 5, 6, 7, 8],
        'C': [9, 10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nData validation: {'Passed' if is_valid else 'Failed'}")