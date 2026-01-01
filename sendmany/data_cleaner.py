
import pandas as pd

def clean_dataset(df, subset=None, fill_method='mean'):
    """
    Clean a pandas DataFrame by removing duplicate rows and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    subset (list, optional): List of column names to consider for duplicate removal.
    fill_method (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates(subset=subset, keep='first')
    
    # Handle missing values
    if fill_method == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif fill_method == 'mean':
        df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
    elif fill_method == 'median':
        df_cleaned = df_cleaned.fillna(df_cleaned.median(numeric_only=True))
    elif fill_method == 'mode':
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else None)
            else:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 0)
    else:
        raise ValueError("fill_method must be 'drop', 'mean', 'median', or 'mode'")
    
    return df_cleaned

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        if df[numeric_cols].isin([float('inf'), float('-inf')]).any().any():
            print("DataFrame contains infinite values in numeric columns.")
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, subset=['A', 'C'], fill_method='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid = validate_data(cleaned_df, required_columns=['A', 'B', 'C'])
    print(f"\nData validation passed: {is_valid}")