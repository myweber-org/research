
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
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def z_score_normalize(data, column):
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

def min_max_normalize(data, column, feature_range=(0, 1)):
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

def detect_missing_patterns(data, threshold=0.3):
    """
    Detect columns with high percentage of missing values
    """
    missing_percentage = data.isnull().sum() / len(data)
    high_missing_cols = missing_percentage[missing_percentage > threshold].index.tolist()
    
    return {
        'missing_percentage': missing_percentage,
        'high_missing_columns': high_missing_cols,
        'total_missing': data.isnull().sum().sum()
    }

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5, normalize_method='zscore'):
    """
    Main function to clean dataset with multiple operations
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    cleaning_report = {
        'original_shape': data.shape,
        'outliers_removed': {},
        'columns_normalized': []
    }
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            try:
                filtered_data, removed = remove_outliers_iqr(cleaned_data, col, outlier_factor)
                if removed > 0:
                    cleaned_data = filtered_data
                    cleaning_report['outliers_removed'][col] = removed
                
                if normalize_method == 'zscore':
                    cleaned_data[f'{col}_normalized'] = z_score_normalize(cleaned_data, col)
                elif normalize_method == 'minmax':
                    cleaned_data[f'{col}_normalized'] = min_max_normalize(cleaned_data, col)
                
                cleaning_report['columns_normalized'].append(col)
                
            except Exception as e:
                print(f"Warning: Could not process column {col}: {str(e)}")
    
    cleaning_report['final_shape'] = cleaned_data.shape
    cleaning_report['missing_info'] = detect_missing_patterns(cleaned_data)
    
    return cleaned_data, cleaning_report

def validate_data_types(data, expected_types):
    """
    Validate that columns have expected data types
    """
    validation_results = {}
    
    for col, expected_type in expected_types.items():
        if col in data.columns:
            actual_type = str(data[col].dtype)
            validation_results[col] = {
                'expected': expected_type,
                'actual': actual_type,
                'valid': expected_type in actual_type
            }
    
    return validation_results