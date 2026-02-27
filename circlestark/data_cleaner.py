
import re
import string

def normalize_text(text):
    """
    Normalize text by converting to lowercase, removing extra whitespace,
    and stripping punctuation from the edges.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Strip punctuation from the beginning and end
    text = text.strip(string.punctuation)
    
    return text

def remove_special_characters(text, keep_spaces=True):
    """
    Remove all non-alphanumeric characters from text.
    
    Args:
        text: Input string to clean
        keep_spaces: If True, preserve spaces between words
    
    Returns:
        Cleaned string containing only alphanumeric characters and optionally spaces
    """
    if not isinstance(text, str):
        return ""
    
    if keep_spaces:
        # Keep letters, numbers, and spaces
        pattern = r'[^a-zA-Z0-9\s]'
    else:
        # Keep only letters and numbers
        pattern = r'[^a-zA-Z0-9]'
    
    return re.sub(pattern, '', text)

def clean_whitespace(text):
    """
    Clean and normalize all whitespace in text.
    Replaces tabs, newlines, and multiple spaces with single spaces.
    """
    if not isinstance(text, str):
        return ""
    
    # Replace all whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def get_word_count(text):
    """
    Count the number of words in a text string.
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    
    # Split by whitespace and count non-empty elements
    words = [word for word in text.split() if word]
    return len(words)

def truncate_text(text, max_length, suffix="..."):
    """
    Truncate text to a maximum length, adding suffix if truncated.
    
    Args:
        text: Input text to truncate
        max_length: Maximum allowed length
        suffix: String to append if text is truncated
    
    Returns:
        Truncated text with suffix if necessary
    """
    if not isinstance(text, str):
        return ""
    
    if len(text) <= max_length:
        return text
    
    # Calculate truncation point accounting for suffix length
    truncate_point = max_length - len(suffix)
    if truncate_point <= 0:
        return suffix
    
    return text[:truncate_point].rstrip() + suffix
import numpy as np
import pandas as pd
from scipy import stats

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
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f'{column}_standardized'] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate summary statistics for the dataset
    """
    summary = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': data.isnull().sum().to_dict(),
        'data_types': data.dtypes.to_dict()
    }
    
    for col in data.select_dtypes(include=[np.number]).columns:
        summary[f'{col}_stats'] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'median': data[col].median()
        }
    
    return summary