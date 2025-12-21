import csv
import re
from typing import List, Dict, Any

def clean_string(value: str) -> str:
    """Remove extra whitespace and convert to lowercase."""
    if not isinstance(value, str):
        return value
    return re.sub(r'\s+', ' ', value.strip()).lower()

def clean_numeric(value: str) -> float:
    """Convert string to float, handling empty values."""
    try:
        return float(value.replace(',', ''))
    except ValueError:
        return 0.0

def read_and_clean_csv(filepath: str) -> List[Dict[str, Any]]:
    """Read CSV file and apply cleaning functions to each row."""
    cleaned_data = []
    
    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            cleaned_row = {}
            for key, value in row.items():
                if key.endswith('_amount') or key.endswith('_price'):
                    cleaned_row[key] = clean_numeric(value)
                else:
                    cleaned_row[key] = clean_string(value)
            cleaned_data.append(cleaned_row)
    
    return cleaned_data

def write_cleaned_csv(data: List[Dict[str, Any]], output_path: str) -> None:
    """Write cleaned data to a new CSV file."""
    if not data:
        return
    
    fieldnames = data[0].keys()
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def remove_duplicates(data: List[Dict[str, Any]], key_field: str) -> List[Dict[str, Any]]:
    """Remove duplicate rows based on a specified key field."""
    seen = set()
    unique_data = []
    
    for row in data:
        identifier = row.get(key_field)
        if identifier not in seen:
            seen.add(identifier)
            unique_data.append(row)
    
    return unique_data

def filter_by_threshold(data: List[Dict[str, Any]], 
                        field: str, 
                        threshold: float) -> List[Dict[str, Any]]:
    """Filter rows where the specified field meets or exceeds threshold."""
    return [row for row in data if row.get(field, 0) >= threshold]
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def process_numerical_columns(df, columns=None):
    """
    Process multiple numerical columns for outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, processes all numerical columns.
    
    Returns:
    pd.DataFrame: Processed DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    processed_df = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            try:
                processed_df = remove_outliers_iqr(processed_df, col)
            except Exception as e:
                print(f"Error processing column {col}: {e}")
    
    return processed_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add some outliers
    df.loc[1000] = [500, 1000, 300]
    df.loc[1001] = [-100, -50, -10]
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics for column 'A':")
    print(calculate_summary_statistics(df, 'A'))
    
    # Process the data
    cleaned_df = process_numerical_columns(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics for column 'A':")
    print(calculate_summary_statistics(cleaned_df, 'A'))import numpy as np
import pandas as pd

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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

def generate_summary(df):
    summary = {
        'original_rows': len(df),
        'cleaned_rows': len(df.dropna()),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict()
    }
    return summary