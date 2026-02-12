
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