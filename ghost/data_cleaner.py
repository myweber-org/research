import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
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
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame is valid"

def process_data_file(file_path, output_path=None, **clean_kwargs):
    """
    Load, clean, and optionally save a data file.
    
    Args:
        file_path (str): Path to input data file.
        output_path (str): Path to save cleaned data.
        **clean_kwargs: Additional arguments for clean_dataframe.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        cleaned_df = clean_dataframe(df, **clean_kwargs)
        
        is_valid, message = validate_dataframe(cleaned_df)
        if not is_valid:
            print(f"Warning: {message}")
        
        if output_path:
            if output_path.endswith('.csv'):
                cleaned_df.to_csv(output_path, index=False)
            elif output_path.endswith('.xlsx'):
                cleaned_df.to_excel(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return cleaned_df
        
    except Exception as e:
        print(f"Error processing file: {e}")
        raise