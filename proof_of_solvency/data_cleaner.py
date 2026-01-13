
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Dropped {len(df) - len(cleaned_df)} duplicate rows.")
    
    missing_before = cleaned_df.isnull().sum().sum()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print(f"Dropped rows with missing values. Removed {missing_before} missing entries.")
    elif fill_missing in ['mean', 'median', 'mode']:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                cleaned_df[column].fillna(fill_value, inplace=True)
        print(f"Filled missing values using {fill_missing} method.")
    
    missing_after = cleaned_df.isnull().sum().sum()
    print(f"Missing values before: {missing_before}, after: {missing_after}")
    
    return cleaned_df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate the DataFrame for required columns and numeric data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    numeric_columns (list): List of column names that must be numeric.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if numeric_columns:
        non_numeric = [col for col in numeric_columns if not np.issubdtype(df[col].dtype, np.number)]
        if non_numeric:
            print(f"Non-numeric columns found: {non_numeric}")
            return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, np.nan],
        'B': [10, np.nan, 30, 40, 50, 60],
        'C': ['x', 'y', 'y', 'z', 'x', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'], numeric_columns=['A', 'B'])
    print(f"\nData validation result: {is_valid}")