import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame using specified strategy.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Method for handling missing values ('mean', 'median', 'mode', 'drop')
    columns (list): Specific columns to clean, if None cleans all columns
    
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
            
        if df_clean[col].isnull().sum() == 0:
            continue
            
        if strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        elif strategy == 'mode':
            if not df_clean[col].empty:
                mode_value = df_clean[col].mode()
                if not mode_value.empty:
                    df_clean[col].fillna(mode_value[0], inplace=True)
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    
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

def load_and_clean_csv(filepath, **kwargs):
    """
    Load CSV file and clean missing data.
    
    Parameters:
    filepath (str): Path to CSV file
    **kwargs: Additional arguments passed to clean_missing_data
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        is_valid, message = validate_dataframe(df)
        
        if not is_valid:
            print(f"Warning: {message}")
            return df
        
        return clean_missing_data(df, **kwargs)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned with mean strategy:")
    print(clean_missing_data(df, strategy='mean'))
    print("\nCleaned with drop strategy:")
    print(clean_missing_data(df, strategy='drop'))