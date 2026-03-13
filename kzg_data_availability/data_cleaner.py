
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column names to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of column to clean
        fill_method (str): Method to fill missing values ('mean', 'median', 'zero')
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(f"Column '{column_name}' is not numeric")
    
    df_copy = df.copy()
    
    if fill_method == 'mean':
        fill_value = df_copy[column_name].mean()
    elif fill_method == 'median':
        fill_value = df_copy[column_name].median()
    elif fill_method == 'zero':
        fill_value = 0
    else:
        raise ValueError("fill_method must be 'mean', 'median', or 'zero'")
    
    missing_count = df_copy[column_name].isna().sum()
    if missing_count > 0:
        df_copy[column_name] = df_copy[column_name].fillna(fill_value)
        print(f"Filled {missing_count} missing values in '{column_name}' with {fill_method}")
    
    return df_copy

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate summary statistics for DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isna().sum().to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return summary
def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        for column in cleaned_df.select_dtypes(include=['number']).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (bool, str) indicating validation result and message.
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame validation passed"
import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Parameters:
    file_path (str): Path to input CSV file
    output_path (str, optional): Path to save cleaned CSV file
    missing_strategy (str): Strategy for handling missing values
                           'mean', 'median', 'mode', or 'drop'
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            if missing_strategy == 'mean':
                df = df.fillna(df.mean(numeric_only=True))
            elif missing_strategy == 'median':
                df = df.fillna(df.median(numeric_only=True))
            elif missing_strategy == 'mode':
                df = df.fillna(df.mode().iloc[0])
            elif missing_strategy == 'drop':
                df = df.dropna()
            else:
                raise ValueError(f"Unknown strategy: {missing_strategy}")
            
            print(f"Missing values handled using '{missing_strategy}' strategy")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_numeric_columns(df, columns=None):
    """
    Validate that specified columns contain only numeric data.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate
    columns (list, optional): List of column names to validate
    
    Returns:
    dict: Validation results for each column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    results = {}
    for col in columns:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
            results[col] = {
                'total_values': len(df[col]),
                'non_numeric_count': non_numeric,
                'is_valid': non_numeric == 0
            }
    
    return results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10.5, np.nan, 30.2, 40.1, 50.0],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 
                               output_path='cleaned_data.csv',
                               missing_strategy='mean')
    
    if cleaned_df is not None:
        validation = validate_numeric_columns(cleaned_df)
        print("Validation results:", validation)
import pandas as pd
import re

def clean_dataframe(df, column_name):
    """
    Clean a specified column in a DataFrame by removing duplicates,
    stripping whitespace, and converting to lowercase.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df_clean = df.copy()
    df_clean[column_name] = df_clean[column_name].astype(str)
    df_clean[column_name] = df_clean[column_name].str.strip()
    df_clean[column_name] = df_clean[column_name].str.lower()
    df_clean = df_clean.drop_duplicates(subset=[column_name], keep='first')
    
    return df_clean

def remove_special_characters(text):
    """
    Remove special characters from a string, keeping only alphanumeric and spaces.
    """
    if pd.isna(text):
        return text
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(text))

def normalize_column(df, column_name):
    """
    Apply special character removal and cleaning to a column.
    """
    df[column_name] = df[column_name].apply(remove_special_characters)
    return clean_dataframe(df, column_name)

if __name__ == "__main__":
    sample_data = {'Name': [' Alice ', 'bob', 'Alice', 'Charlie!', '  david  ']}
    df_sample = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df_sample)
    
    cleaned_df = normalize_column(df_sample, 'Name')
    print("\nCleaned DataFrame:")
    print(cleaned_df)