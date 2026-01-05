
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list, optional): List of column names to check for duplicates.
                                       If None, checks all columns.
    fill_missing (str, optional): Strategy to fill missing values.
                                  Options: 'mean', 'median', 'mode', 'drop', or a numeric value.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    original_shape = df.shape
    
    # Remove duplicates
    if columns_to_check is None:
        df_cleaned = df.drop_duplicates()
    else:
        df_cleaned = df.drop_duplicates(subset=columns_to_check)
    
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    
    # Handle missing values
    if fill_missing == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if fill_missing == 'mean':
                fill_value = df_cleaned[col].mean()
            elif fill_missing == 'median':
                fill_value = df_cleaned[col].median()
            elif fill_missing == 'mode':
                fill_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 0
            
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    elif isinstance(fill_missing, (int, float)):
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(fill_missing)
    
    missing_filled = df.isna().sum().sum() - df_cleaned.isna().sum().sum()
    
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Missing values filled: {missing_filled}")
    
    return df_cleaned

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of required column names.
    min_rows (int, optional): Minimum number of rows required.
    
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

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0, np.nan],
        'category': ['A', 'B', 'B', 'C', 'A', 'B', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, columns_to_check=['id'], fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid = validate_data(cleaned_df, required_columns=['id', 'value'], min_rows=1)
    print(f"\nData validation passed: {is_valid}")