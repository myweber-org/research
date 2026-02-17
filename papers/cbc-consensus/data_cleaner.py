
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    
    if len(filtered_indices) == 0:
        return data.iloc[[]]
    
    filtered_data = data.iloc[filtered_indices]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data_copy = data.copy()
    min_val = data_copy[column].min()
    max_val = data_copy[column].max()
    
    if max_val == min_val:
        data_copy[column] = 0.5
    else:
        data_copy[column] = (data_copy[column] - min_val) / (max_val - min_val)
    
    return data_copy

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data_copy = data.copy()
    mean_val = data_copy[column].mean()
    std_val = data_copy[column].std()
    
    if std_val == 0:
        data_copy[column] = 0
    else:
        data_copy[column] = (data_copy[column] - mean_val) / std_val
    
    return data_copy

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names to process
        outlier_method: 'iqr' or 'zscore' (default 'iqr')
        normalize_method: 'minmax' or 'zscore' (default 'minmax')
    
    Returns:
        Cleaned DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError("outlier_method must be 'iqr' or 'zscore'")
        
        if normalize_method == 'minmax':
            cleaned_data = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError("normalize_method must be 'minmax' or 'zscore'")
    
    return cleaned_data

def validate_data(data, required_columns, numeric_columns):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        numeric_columns: list of numeric column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    for column in numeric_columns:
        if column in data.columns and not pd.api.types.is_numeric_dtype(data[column]):
            return False, f"Column '{column}' is not numeric"
    
    if data.empty:
        return False, "DataFrame is empty"
    
    return True, "Data validation passed"
import csv
import sys

def remove_duplicates(input_file, output_file, key_column):
    """
    Remove duplicate rows from a CSV file based on a specified key column.
    """
    seen = set()
    unique_rows = []
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            
            if key_column not in fieldnames:
                raise ValueError(f"Key column '{key_column}' not found in CSV headers")
            
            for row in reader:
                key_value = row[key_column]
                if key_value not in seen:
                    seen.add(key_value)
                    unique_rows.append(row)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(unique_rows)
        
        print(f"Removed {len(seen) - len(unique_rows)} duplicate rows")
        print(f"Unique rows: {len(unique_rows)}")
        print(f"Output saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python data_cleaner.py <input_file> <output_file> <key_column>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key_column = sys.argv[3]
    
    remove_duplicates(input_file, output_file, key_column)