import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling to range [0, 1].
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        normalized = pd.Series([0.5] * len(data), index=data.index)
    else:
        normalized = (data[column] - min_val) / (max_val - min_val)
    
    return normalized

def z_score_normalize(data, column):
    """
    Normalize data using Z-score standardization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        normalized = pd.Series([0] * len(data), index=data.index)
    else:
        normalized = (data[column] - mean_val) / std_val
    
    return normalized

def clean_dataset(data, numeric_columns, outlier_factor=1.5, normalization_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_data = data.copy()
    removal_stats = {}
    
    for col in numeric_columns:
        if col not in cleaned_data.columns:
            continue
            
        cleaned_data, removed = remove_outliers_iqr(cleaned_data, col, outlier_factor)
        removal_stats[col] = removed
        
        if normalization_method == 'minmax':
            cleaned_data[f'{col}_normalized'] = normalize_minmax(cleaned_data, col)
        elif normalization_method == 'zscore':
            cleaned_data[f'{col}_normalized'] = z_score_normalize(cleaned_data, col)
    
    return cleaned_data, removal_stats

def validate_data(data, required_columns, numeric_threshold=0.8):
    """
    Validate dataset structure and content.
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    numeric_ratio = data.select_dtypes(include=[np.number]).shape[1] / data.shape[1]
    
    if numeric_ratio < numeric_threshold:
        return False, f"Numeric columns ratio ({numeric_ratio:.2f}) below threshold ({numeric_threshold})"
    
    return True, "Dataset validation passed"