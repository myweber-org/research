
import re
import pandas as pd
from typing import Union, List, Optional

def remove_duplicates(data: Union[List, pd.Series, pd.DataFrame]) -> Union[List, pd.Series, pd.DataFrame]:
    """Remove duplicate entries from a list, Series, or DataFrame."""
    if isinstance(data, list):
        return list(dict.fromkeys(data))
    elif isinstance(data, pd.Series):
        return data.drop_duplicates()
    elif isinstance(data, pd.DataFrame):
        return data.drop_duplicates()
    else:
        raise TypeError("Input must be a list, pandas Series, or pandas DataFrame")

def validate_email(email: str) -> bool:
    """Validate an email address format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def normalize_string(text: str, case: str = 'lower') -> str:
    """Normalize string by converting to specified case and stripping whitespace."""
    text = text.strip()
    if case == 'lower':
        return text.lower()
    elif case == 'upper':
        return text.upper()
    elif case == 'title':
        return text.title()
    else:
        return text

def fill_missing_values(df: pd.DataFrame, column: str, value: Union[str, int, float]) -> pd.DataFrame:
    """Fill missing values in a DataFrame column with a specified value."""
    df_copy = df.copy()
    df_copy[column] = df_copy[column].fillna(value)
    return df_copy

def filter_by_threshold(data: List[float], threshold: float) -> List[float]:
    """Filter a list of floats, keeping values above a specified threshold."""
    return [x for x in data if x > threshold]

def calculate_statistics(numbers: List[float]) -> dict:
    """Calculate basic statistics from a list of numbers."""
    if not numbers:
        return {'mean': None, 'median': None, 'min': None, 'max': None}
    
    sorted_nums = sorted(numbers)
    n = len(numbers)
    
    return {
        'mean': sum(numbers) / n,
        'median': sorted_nums[n // 2] if n % 2 != 0 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2,
        'min': min(numbers),
        'max': max(numbers)
    }

def sanitize_filename(filename: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized.strip('_.')

def convert_to_boolean(value: Union[str, int, bool]) -> Optional[bool]:
    """Convert various representations to boolean."""
    if isinstance(value, bool):
        return value
    elif isinstance(value, (int, float)):
        return bool(value)
    elif isinstance(value, str):
        lower_val = value.lower().strip()
        if lower_val in ('true', 'yes', 'y', '1', 't'):
            return True
        elif lower_val in ('false', 'no', 'n', '0', 'f'):
            return False
    return None

def split_camel_case(text: str) -> str:
    """Split camelCase or PascalCase strings into separate words."""
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

def validate_phone_number(phone: str) -> bool:
    """Validate a phone number format (basic international format)."""
    pattern = r'^\+?[1-9]\d{1,14}$'
    cleaned = re.sub(r'[\s\-()]', '', phone)
    return bool(re.match(pattern, cleaned))
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
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
    
    if max_val == min_val:
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

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='zscore'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns
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
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate data structure and content
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan and data.isnull().any().any():
        raise ValueError("Data contains NaN values")
    
    return True