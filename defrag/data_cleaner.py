import csv
import re
from typing import List, Dict, Any, Optional

def clean_csv_data(input_file: str, output_file: str, columns_to_clean: Optional[List[str]] = None) -> None:
    """
    Clean data in a CSV file by removing extra whitespace and standardizing text.
    """
    cleaned_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            cleaned_row = {}
            for key, value in row.items():
                if columns_to_clean is None or key in columns_to_clean:
                    if isinstance(value, str):
                        # Remove extra whitespace
                        cleaned_value = re.sub(r'\s+', ' ', value.strip())
                        # Convert to title case for consistency
                        cleaned_value = cleaned_value.title()
                    else:
                        cleaned_value = value
                else:
                    cleaned_value = value
                cleaned_row[key] = cleaned_value
            cleaned_rows.append(cleaned_row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

def validate_email_format(email: str) -> bool:
    """
    Validate if a string is in a proper email format.
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))

def filter_invalid_emails(input_file: str, email_column: str) -> List[Dict[str, Any]]:
    """
    Filter rows from a CSV file where the email column contains invalid email addresses.
    """
    valid_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        
        for row in reader:
            if email_column in row and validate_email_format(row[email_column]):
                valid_rows.append(row)
    
    return valid_rows

def remove_duplicates(data: List[Dict[str, Any]], key_column: str) -> List[Dict[str, Any]]:
    """
    Remove duplicate rows based on a specified key column.
    """
    seen = set()
    unique_data = []
    
    for row in data:
        key_value = row.get(key_column)
        if key_value not in seen:
            seen.add(key_value)
            unique_data.append(row)
    
    return unique_dataimport pandas as pd

def clean_dataset(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_na (bool): Whether to drop rows with null values
        rename_columns (bool): Whether to standardize column names
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if drop_na:
        df_clean = df_clean.dropna()
    
    if rename_columns:
        df_clean.columns = (
            df_clean.columns
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^a-z0-9_]', '', regex=True)
        )
    
    return df_clean

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Dataset is valid"

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list): Columns to consider for duplicates
        keep (str): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)