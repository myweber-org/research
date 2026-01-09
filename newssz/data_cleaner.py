import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str): Strategy to fill missing values. 
                        Options: 'mean', 'median', 'mode', or 'drop'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
        print(f"Removed {len(df) - len(df_clean)} duplicate rows.")
    
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values.")
        
        if fill_missing == 'drop':
            df_clean = df_clean.dropna()
            print("Dropped rows with missing values.")
        elif fill_missing in ['mean', 'median']:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if fill_missing == 'mean':
                    fill_value = df_clean[col].mean()
                else:
                    fill_value = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(fill_value)
            print(f"Filled missing numeric values with {fill_missing}.")
        elif fill_missing == 'mode':
            for col in df_clean.columns:
                mode_value = df_clean[col].mode()
                if not mode_value.empty:
                    df_clean[col] = df_clean[col].fillna(mode_value.iloc[0])
            print("Filled missing values with mode.")
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {df_clean.shape}")
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if DataFrame passes validation.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        print("Warning: DataFrame is empty.")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, np.nan],
        'B': [5, np.nan, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_dataframe(cleaned_df, required_columns=['A', 'B'])
        print("Data validation passed.")
    except Exception as e:
        print(f"Data validation failed: {e}")