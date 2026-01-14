import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Load a CSV file, handle missing values, and save cleaned data.
    
    Args:
        input_file (str): Path to input CSV file.
        output_file (str): Path to save cleaned CSV file.
        missing_strategy (str): Strategy for handling missing values.
            Options: 'mean', 'median', 'drop', 'zero'.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif missing_strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif missing_strategy == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        elif missing_strategy == 'drop':
            df = df.dropna(subset=numeric_cols)
        else:
            raise ValueError(f"Unknown strategy: {missing_strategy}")
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data shape: {df.shape}")
        print(f"Cleaned data saved to: {output_file}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df, column_name, min_value=None, max_value=None):
    """
    Validate data in a specific column.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        column_name (str): Column to check.
        min_value (float): Minimum allowed value.
        max_value (float): Maximum allowed value.
    
    Returns:
        bool: True if validation passes.
    """
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found.")
        return False
    
    if min_value is not None:
        if (df[column_name] < min_value).any():
            print(f"Values below minimum {min_value} found in '{column_name}'.")
            return False
    
    if max_value is not None:
        if (df[column_name] > max_value).any():
            print(f"Values above maximum {max_value} found in '{column_name}'.")
            return False
    
    return True

if __name__ == "__main__":
    cleaned_df = clean_csv_data('raw_data.csv', 'cleaned_data.csv', 'mean')
    
    if cleaned_df is not None:
        is_valid = validate_data(cleaned_df, 'temperature', min_value=-50, max_value=100)
        if is_valid:
            print("Data validation passed.")
        else:
            print("Data validation failed.")