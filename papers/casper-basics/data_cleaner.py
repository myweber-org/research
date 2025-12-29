import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values with column mean.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                mean_val = cleaned_df[col].mean()
                cleaned_df[col] = cleaned_df[col].fillna(mean_val)
                print(f"Filled missing values in '{col}' with mean: {mean_val:.2f}")
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.empty:
        print("DataFrame is empty.")
        return False
    
    return True

def process_data_file(file_path, output_path=None):
    """
    Load, clean, and save a dataset from a CSV file.
    
    Args:
        file_path (str): Path to input CSV file.
        output_path (str): Path to save cleaned CSV file.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}. Shape: {df.shape}")
        
        if validate_data(df):
            cleaned_df = clean_dataset(df)
            print(f"Cleaned data shape: {cleaned_df.shape}")
            
            if output_path:
                cleaned_df.to_csv(output_path, index=False)
                print(f"Saved cleaned data to {output_path}")
            
            return cleaned_df
        else:
            print("Data validation failed.")
            return None
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, None, 15.2, 20.1, None],
        'category': ['A', 'B', 'B', 'A', 'C']
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned = clean_dataset(sample_data)
    print("\nCleaned data:")
    print(cleaned)