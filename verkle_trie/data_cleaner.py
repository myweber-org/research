def remove_duplicates(input_list):
    """
    Remove duplicate items from a list while preserving the original order.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_strings(string_list):
    """
    Clean a list of strings by converting numeric strings to integers.
    Non-numeric strings are kept as-is.
    """
    cleaned = []
    for s in string_list:
        s_stripped = s.strip()
        if s_stripped.isdigit():
            cleaned.append(int(s_stripped))
        else:
            cleaned.append(s)
    return cleaned

def filter_by_type(data_list, data_type):
    """
    Filter a list to include only items of a specific type.
    """
    return [item for item in data_list if isinstance(item, data_type)]

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, "4", "4", "five", 5.0, 5.0]
    print("Original:", sample_data)
    
    unique_data = remove_duplicates(sample_data)
    print("After removing duplicates:", unique_data)
    
    cleaned_data = clean_numeric_strings(unique_data)
    print("After cleaning numeric strings:", cleaned_data)
    
    integers_only = filter_by_type(cleaned_data, int)
    print("Integers only:", integers_only)import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, columns=None):
    """
    Normalize data using min-max scaling
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val > min_val:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
    
    return normalized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    cleaned_data = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if data[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(data[col]):
                cleaned_data[col].fillna(data[col].mean(), inplace=True)
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(data[col]):
                cleaned_data[col].fillna(data[col].median(), inplace=True)
            elif strategy == 'mode':
                cleaned_data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 0, inplace=True)
            elif strategy == 'drop':
                cleaned_data = cleaned_data.dropna(subset=[col])
    
    return cleaned_data

def clean_dataset(data, config):
    """
    Main cleaning pipeline
    """
    cleaned_data = data.copy()
    
    if 'outlier_columns' in config:
        for col in config['outlier_columns']:
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    if 'missing_strategy' in config:
        cleaned_data = handle_missing_values(
            cleaned_data, 
            strategy=config['missing_strategy'],
            columns=config.get('missing_columns')
        )
    
    if 'normalize_columns' in config:
        cleaned_data = normalize_minmax(cleaned_data, config['normalize_columns'])
    
    return cleaned_data