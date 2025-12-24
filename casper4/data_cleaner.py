import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def zscore_normalize(data, column):
    """
    Normalize data using z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column]
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def minmax_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column]
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    
    if feature_range != (0, 1):
        new_min, new_max = feature_range
        normalized = normalized * (new_max - new_min) + new_min
    
    return normalized

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5, normalize_method='zscore'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    stats_report = {}
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        original_count = len(cleaned_df)
        cleaned_df, outliers_removed = remove_outliers_iqr(cleaned_df, col, outlier_factor)
        
        if normalize_method == 'zscore':
            cleaned_df[f'{col}_normalized'] = zscore_normalize(cleaned_df, col)
        elif normalize_method == 'minmax':
            cleaned_df[f'{col}_normalized'] = minmax_normalize(cleaned_df, col)
        else:
            raise ValueError("normalize_method must be 'zscore' or 'minmax'")
        
        stats_report[col] = {
            'original_samples': original_count,
            'cleaned_samples': len(cleaned_df),
            'outliers_removed': outliers_removed,
            'normalization_method': normalize_method
        }
    
    return cleaned_df, stats_report

def validate_data(df, required_columns, min_rows=10):
    """
    Validate dataset structure and content
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(df) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows, got {len(df)}")
    
    null_counts = df[required_columns].isnull().sum()
    high_null_columns = null_counts[null_counts > 0.1 * len(df)].index.tolist()
    
    validation_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_columns': missing_columns,
        'high_null_columns': high_null_columns,
        'null_counts': null_counts.to_dict()
    }
    
    return validation_report