
import pandas as pd
import numpy as np

def clean_missing_data(df, method='mean', columns=None):
    """
    Clean missing values in a DataFrame using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    method (str): Cleaning method - 'mean', 'median', 'mode', or 'drop'
    columns (list): Specific columns to clean, None for all columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if method == 'mean':
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        elif method == 'median':
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        elif method == 'mode':
            if not df_clean[col].mode().empty:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        elif method == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def load_and_clean_csv(filepath, cleaning_method='mean'):
    """
    Load CSV file and clean missing values.
    
    Parameters:
    filepath (str): Path to CSV file
    cleaning_method (str): Method for cleaning missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        is_valid, message = validate_dataframe(df)
        
        if not is_valid:
            print(f"Validation failed: {message}")
            return pd.DataFrame()
        
        df_clean = clean_missing_data(df, method=cleaning_method)
        print(f"Cleaned {df.isnull().sum().sum()} missing values using {cleaning_method} method")
        return df_clean
        
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_missing_data(df, method='mean')
    print("\nCleaned DataFrame (mean imputation):")
    print(cleaned_df)