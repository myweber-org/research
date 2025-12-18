import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (bool): If True, fill missing values with fill_value.
    fill_value: Value to use for filling missing values if fill_missing is True.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns is None:
        required_columns = []
    
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return False
    
    return True

def process_data(file_path, output_path=None):
    """
    Load, clean, and save data from a CSV file.
    
    Parameters:
    file_path (str): Path to input CSV file.
    output_path (str): Path to save cleaned data. If None, returns DataFrame.
    
    Returns:
    pd.DataFrame or None: Cleaned DataFrame if output_path is None.
    """
    try:
        df = pd.read_csv(file_path)
        
        if validate_dataframe(df):
            cleaned_df = clean_dataframe(df, drop_duplicates=True, fill_missing=True)
            
            if output_path:
                cleaned_df.to_csv(output_path, index=False)
                print(f"Data saved to {output_path}")
                return None
            else:
                return cleaned_df
        else:
            print("Data validation failed")
            return None
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None