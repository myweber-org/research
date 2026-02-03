import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
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

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers from a specific column using z-score method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    threshold (float): Z-score threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    from scipy import stats
    import numpy as np
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    mask = z_scores < threshold
    
    valid_indices = df[column].dropna().index[mask]
    return df.loc[valid_indices].reset_index(drop=True)import csv
import re
from typing import List, Dict, Optional

def remove_duplicates(data: List[Dict]) -> List[Dict]:
    """Remove duplicate rows based on all column values."""
    seen = set()
    unique_data = []
    
    for row in data:
        row_tuple = tuple(sorted(row.items()))
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_data.append(row)
    
    return unique_data

def clean_numeric(value: str) -> Optional[float]:
    """Extract numeric values from strings containing symbols and text."""
    if not value or not isinstance(value, str):
        return None
    
    numeric_match = re.search(r'[-+]?\d*\.?\d+', value)
    if numeric_match:
        try:
            return float(numeric_match.group())
        except ValueError:
            return None
    return None

def standardize_phone(phone: str) -> Optional[str]:
    """Standardize phone number format to (XXX) XXX-XXXX."""
    if not phone:
        return None
    
    digits = re.sub(r'\D', '', phone)
    
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits[0] == '1':
        return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    
    return None

def validate_email(email: str) -> bool:
    """Basic email validation using regex pattern."""
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def load_csv(filepath: str) -> List[Dict]:
    """Load CSV file and return list of dictionaries."""
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return list(reader)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return []
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

def save_csv(data: List[Dict], filepath: str) -> bool:
    """Save list of dictionaries to CSV file."""
    if not data:
        return False
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return False

def clean_csv_data(input_file: str, output_file: str) -> None:
    """Main function to clean CSV data with multiple cleaning operations."""
    data = load_csv(input_file)
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    print(f"Loaded {len(data)} rows from {input_file}")
    
    cleaned_data = []
    for row in data:
        cleaned_row = row.copy()
        
        for key, value in row.items():
            if key.lower().endswith('phone'):
                cleaned_row[key] = standardize_phone(value)
            elif key.lower().endswith('email'):
                if not validate_email(value):
                    cleaned_row[key] = None
            elif any(num_term in key.lower() for num_term in ['amount', 'price', 'quantity']):
                cleaned_row[key] = clean_numeric(value)
        
        cleaned_data.append(cleaned_row)
    
    cleaned_data = remove_duplicates(cleaned_data)
    
    if save_csv(cleaned_data, output_file):
        print(f"Saved {len(cleaned_data)} cleaned rows to {output_file}")
    else:
        print("Failed to save cleaned data.")