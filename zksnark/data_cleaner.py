import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize column using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def clean_dataset(df, numeric_columns, outlier_multiplier=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns
    """
    original_shape = df.shape
    
    for col in numeric_columns:
        if col in df.columns:
            df, removed = remove_outliers_iqr(df, col, outlier_multiplier)
            print(f"Removed {removed} outliers from column '{col}'")
    
    for col in numeric_columns:
        if col in df.columns:
            df[f"{col}_normalized"] = normalize_minmax(df, col)
    
    print(f"Dataset cleaned: {original_shape[0]} -> {df.shape[0]} rows")
    return df

def validate_dataframe(df, required_columns):
    """
    Validate DataFrame structure and content
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"Warning: Found null values:\n{null_counts[null_counts > 0]}")
    
    return True