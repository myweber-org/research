import csv
import re
from typing import List, Dict, Any

def remove_duplicates(data: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = set()
    unique_data = []
    for row in data:
        if row[key] not in seen:
            seen.add(row[key])
            unique_data.append(row)
    return unique_data

def normalize_string(value: str) -> str:
    if not isinstance(value, str):
        return value
    value = value.strip().lower()
    value = re.sub(r'\s+', ' ', value)
    return value

def clean_numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r'[^\d.-]', '', value)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0

def read_csv_file(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

def write_csv_file(data: List[Dict[str, Any]], filepath: str, fieldnames: List[str]) -> None:
    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def clean_csv_data(input_file: str, output_file: str, unique_key: str) -> None:
    data = read_csv_file(input_file)
    
    for row in data:
        for key, value in row.items():
            if isinstance(value, str):
                row[key] = normalize_string(value)
            elif key.endswith('_amount') or key.endswith('_price'):
                row[key] = clean_numeric(value)
    
    cleaned_data = remove_duplicates(data, unique_key)
    
    if cleaned_data:
        fieldnames = list(cleaned_data[0].keys())
        write_csv_file(cleaned_data, output_file, fieldnames)
        print(f"Cleaned data saved to {output_file}")
    else:
        print("No data to save")

if __name__ == "__main__":
    clean_csv_data("raw_data.csv", "cleaned_data.csv", "id")import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str or dict): Method to fill missing values.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        else:
            cleaned_df = cleaned_df.fillna(fill_missing)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df.empty:
        return False
    
    if len(df) < min_rows:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False
    
    return True

def process_data(file_path, output_path=None, **clean_kwargs):
    """
    Load, clean, and optionally save data.
    
    Parameters:
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
        
        if not validate_dataframe(df):
            raise ValueError("Data validation failed")
        
        cleaned_df = clean_dataframe(df, **clean_kwargs)
        
        if output_path:
            if output_path.endswith('.csv'):
                cleaned_df.to_csv(output_path, index=False)
            elif output_path.endswith('.xlsx'):
                cleaned_df.to_excel(output_path, index=False)
        
        return cleaned_df
    
    except Exception as e:
        print(f"Error processing data: {e}")
        return None