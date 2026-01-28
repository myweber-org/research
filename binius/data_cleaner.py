
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        columns_to_check (list, optional): Specific columns to check for duplicates
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Remove duplicate rows
    if columns_to_check:
        df_cleaned = df.drop_duplicates(subset=columns_to_check)
    else:
        df_cleaned = df.drop_duplicates()
    
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    
    # Handle missing values
    if fill_missing == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                fill_value = df_cleaned[col].mean()
            else:
                fill_value = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    elif fill_missing == 'mode':
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                mode_value = df_cleaned[col].mode()
                if not mode_value.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
    
    missing_filled = df.isna().sum().sum() - df_cleaned.isna().sum().sum()
    
    print(f"Cleaning complete:")
    print(f"  - Removed {duplicates_removed} duplicate rows")
    print(f"  - Filled {missing_filled} missing values")
    print(f"  - Original shape: {original_shape}")
    print(f"  - Cleaned shape: {df_cleaned.shape}")
    
    return df_cleaned

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of columns that must be present
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: Less than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', None],
        'age': [25, 30, 30, None, 35, 40],
        'score': [85.5, 92.0, 92.0, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'], fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid = validate_data(cleaned_df, required_columns=['id', 'name', 'age'], min_rows=3)
    print(f"\nData validation: {'PASSED' if is_valid else 'FAILED'}")