
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Strategy to fill missing values. 
                       Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values.")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values.")
        elif fill_missing == 'mean':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
            print("Filled missing numeric values with column mean.")
        elif fill_missing == 'median':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
            print("Filled missing numeric values with column median.")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
            print("Filled missing categorical values with column mode.")
        else:
            raise ValueError("Invalid fill_missing option. Choose from 'mean', 'median', 'mode', or 'drop'.")
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate basic DataFrame properties.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 5],
        'B': [10, 20, 20, None, 50, 60],
        'C': ['x', 'y', 'y', 'z', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)