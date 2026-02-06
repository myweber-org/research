
import csv
import os
from typing import List, Dict, Any, Optional

def read_csv_file(file_path: str) -> List[Dict[str, Any]]:
    """Read a CSV file and return a list of dictionaries."""
    data = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")
    return data

def clean_numeric_field(record: Dict[str, Any], field: str, default: Any = 0) -> None:
    """Clean a numeric field in a record, converting to float if possible."""
    if field in record:
        try:
            record[field] = float(record[field])
        except (ValueError, TypeError):
            record[field] = default

def remove_empty_records(data: List[Dict[str, Any]], required_fields: List[str]) -> List[Dict[str, Any]]:
    """Remove records that have empty values in required fields."""
    cleaned_data = []
    for record in data:
        keep = True
        for field in required_fields:
            if field not in record or record[field] in (None, "", " "):
                keep = False
                break
        if keep:
            cleaned_data.append(record)
    return cleaned_data

def write_csv_file(data: List[Dict[str, Any]], file_path: str) -> bool:
    """Write data to a CSV file."""
    if not data:
        print("No data to write.")
        return False
    
    try:
        fieldnames = data[0].keys()
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

def clean_csv_data(input_path: str, output_path: str, numeric_fields: Optional[List[str]] = None, required_fields: Optional[List[str]] = None) -> None:
    """Main function to clean CSV data."""
    if numeric_fields is None:
        numeric_fields = []
    if required_fields is None:
        required_fields = []
    
    print(f"Reading data from {input_path}")
    data = read_csv_file(input_path)
    
    if not data:
        print("No data loaded.")
        return
    
    print(f"Loaded {len(data)} records.")
    
    for record in data:
        for field in numeric_fields:
            clean_numeric_field(record, field)
    
    cleaned_data = remove_empty_records(data, required_fields)
    print(f"After cleaning, {len(cleaned_data)} records remain.")
    
    if write_csv_file(cleaned_data, output_path):
        print(f"Cleaned data written to {output_path}")
    else:
        print("Failed to write cleaned data.")
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
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

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame for required columns and data types.
    
    Parameters:
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
        print("DataFrame is empty")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'])
    print(f"\nData validation passed: {is_valid}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    if fill_missing:
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                    df[column].fillna(df[column].mean(), inplace=True)
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                    df[column].fillna(df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    df[column].fillna(df[column].mode()[0], inplace=True)
                elif fill_missing == 'zero':
                    df[column].fillna(0, inplace=True)
                else:
                    df[column].fillna(method='ffill', inplace=True)
    
    print(f"Dataset cleaned: {original_shape} -> {df.shape}")
    return df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate the dataset structure and content.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if numeric_columns:
        for column in numeric_columns:
            if column in df.columns and not pd.api.types.is_numeric_dtype(df[column]):
                try:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                except:
                    raise ValueError(f"Column {column} cannot be converted to numeric")
    
    return True

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column {column} is not numeric")
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        mask = z_scores < threshold
    else:
        raise ValueError(f"Unknown method: {method}")
    
    original_count = len(df)
    df_clean = df[mask].reset_index(drop=True)
    removed_count = original_count - len(df_clean)
    
    print(f"Removed {removed_count} outliers from column '{column}' using {method} method")
    return df_clean

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5, 5, 6],
        'value': [10, 20, 20, None, 50, 1000, 60, None],
        'category': ['A', 'B', 'B', 'C', 'A', 'B', 'B', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    df_clean = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(df_clean)
    
    # Remove outliers
    df_no_outliers = remove_outliers(df_clean, 'value', method='iqr')
    print("\nDataset after outlier removal:")
    print(df_no_outliers)
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(filepath):
    df = pd.read_csv(filepath)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
        df = normalize_minmax(df, col)
    
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv')
    cleaned_df.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaning complete. Original shape: {pd.read_csv('raw_data.csv').shape}, Cleaned shape: {cleaned_df.shape}")