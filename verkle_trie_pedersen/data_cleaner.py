
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = original_shape[0] - cleaned_df.shape[0]
        print(f"Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values
    missing_before = cleaned_df.isnull().sum().sum()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            mode_value = cleaned_df[col].mode()
            if not mode_value.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_value[0])
    
    missing_after = cleaned_df.isnull().sum().sum()
    print(f"Handled {missing_before - missing_after} missing values")
    
    # Final statistics
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
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
    
    print("Data validation passed")
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'A': [1, 2, 2, 3, None, 5],
#         'B': [10, 20, 20, None, 50, 60],
#         'C': ['x', 'y', 'y', 'z', None, 'x']
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     
#     # Clean the data
#     cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
#     
#     # Validate the cleaned data
#     is_valid = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
#     
#     if is_valid:
#         print("\nCleaned DataFrame is ready for analysis")
#         print(cleaned)
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result