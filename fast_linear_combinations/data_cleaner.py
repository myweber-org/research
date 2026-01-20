import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. 
                        Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (mean imputation):")
    cleaned = clean_dataset(df, fill_missing='mean')
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid} - {message}")
import pandas as pd
import numpy as np
from datetime import datetime

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    
    df.drop_duplicates(inplace=True)
    
    df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')
    
    df['numeric_column'] = pd.to_numeric(df['numeric_column'], errors='coerce')
    df['numeric_column'].fillna(df['numeric_column'].mean(), inplace=True)
    
    df['text_column'] = df['text_column'].str.strip().str.lower()
    
    df = df.dropna(subset=['required_column'])
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')
    print(f"Dataset cleaned. Remaining rows: {len(cleaned_df)}")
import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='drop', fill_value=None):
    """
    Load and clean CSV data by handling missing values.
    
    Args:
        filepath (str): Path to the CSV file.
        missing_strategy (str): Strategy for handling missing values.
            Options: 'drop', 'fill', 'ignore'. Default is 'drop'.
        fill_value: Value to fill missing entries if strategy is 'fill'.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    original_rows = len(df)
    
    if missing_strategy == 'drop':
        df_cleaned = df.dropna()
        removed = original_rows - len(df_cleaned)
        print(f"Removed {removed} rows with missing values.")
    
    elif missing_strategy == 'fill':
        if fill_value is None:
            fill_value = df.select_dtypes(include=[np.number]).mean().mean()
        df_cleaned = df.fillna(fill_value)
        print(f"Filled missing values with: {fill_value}")
    
    elif missing_strategy == 'ignore':
        df_cleaned = df
        print("Missing values preserved.")
    
    else:
        raise ValueError("Invalid missing_strategy. Use 'drop', 'fill', or 'ignore'.")
    
    df_cleaned = df_cleaned.reset_index(drop=True)
    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pandas.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if validation passes.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        print("Warning: DataFrame is empty.")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, 7, 8],
        'C': [9, 10, 11, 12]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned = clean_csv_data('test_data.csv', missing_strategy='fill', fill_value=0)
    print("Cleaned DataFrame:")
    print(cleaned)
    
    validation_result = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"Validation passed: {validation_result}")def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data(data):
    if not isinstance(data, list):
        raise TypeError("Input must be a list")
    return remove_duplicates(data)import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fillna_strategy='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if fillna_strategy:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        if fillna_strategy == 'mean':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        elif fillna_strategy == 'median':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        elif fillna_strategy == 'zero':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)
        print(f"Filled missing values in numeric columns using {fillna_strategy}.")
    
    object_cols = cleaned_df.select_dtypes(include=['object']).columns
    cleaned_df[object_cols] = cleaned_df[object_cols].fillna('Unknown')
    print("Filled missing values in object columns with 'Unknown'.")
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame for required columns and basic integrity.
    """
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    if df.empty:
        raise ValueError("DataFrame is empty.")
    
    print("Data validation passed.")
    return True