
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values with different strategies
    """
    cleaned_data = data.copy()
    
    for column in cleaned_data.select_dtypes(include=[np.number]).columns:
        if cleaned_data[column].isnull().any():
            if strategy == 'mean':
                fill_value = cleaned_data[column].mean()
            elif strategy == 'median':
                fill_value = cleaned_data[column].median()
            elif strategy == 'mode':
                fill_value = cleaned_data[column].mode()[0]
            else:
                fill_value = 0
            
            cleaned_data[column] = cleaned_data[column].fillna(fill_value)
    
    return cleaned_data

def validate_dataframe(data):
    """
    Validate dataframe structure and content
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def clean_dataset(data, outlier_method='zscore', normalize_method=None, missing_strategy='mean'):
    """
    Main function to clean dataset with multiple steps
    """
    # Validate input
    validate_dataframe(data)
    
    # Handle missing values
    cleaned_data = handle_missing_values(data, strategy=missing_strategy)
    
    # Remove outliers for numeric columns
    numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, col)
        elif outlier_method == 'iqr':
            outliers = detect_outliers_iqr(cleaned_data, col)
            cleaned_data = cleaned_data.drop(outliers.index)
    
    # Normalize data if requested
    if normalize_method:
        for col in numeric_cols:
            if normalize_method == 'minmax':
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
            elif normalize_method == 'zscore':
                cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    # Reset index after cleaning
    cleaned_data = cleaned_data.reset_index(drop=True)
    
    return cleaned_data