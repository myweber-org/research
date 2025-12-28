
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for identifying duplicates
        keep: determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: If True, fill missing values with fill_value
        fill_value: Value to use for filling missing data
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def calculate_statistics(df, column_name):
    """
    Calculate basic statistics for a specific column.
    
    Args:
        df: pandas DataFrame
        column_name: Name of the column to analyze
    
    Returns:
        Dictionary containing statistics
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    column_data = df[column_name]
    
    stats = {
        'mean': column_data.mean(),
        'median': column_data.median(),
        'std': column_data.std(),
        'min': column_data.min(),
        'max': column_data.max(),
        'count': column_data.count(),
        'missing': column_data.isnull().sum()
    }
    
    return stats

def filter_outliers(df, column_name, method='iqr', threshold=1.5):
    """
    Filter outliers from a DataFrame column.
    
    Args:
        df: pandas DataFrame
        column_name: Name of the column to filter
        method: Method for outlier detection ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    column_data = df[column_name].dropna()
    
    if method == 'iqr':
        Q1 = column_data.quantile(0.25)
        Q3 = column_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    
    elif method == 'zscore':
        mean = column_data.mean()
        std = column_data.std()
        z_scores = abs((df[column_name] - mean) / std)
        filtered_df = df[z_scores <= threshold]
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return filtered_df

def normalize_column(df, column_name, method='minmax'):
    """
    Normalize a column in the DataFrame.
    
    Args:
        df: pandas DataFrame
        column_name: Name of the column to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    normalized_df = df.copy()
    column_data = normalized_df[column_name]
    
    if method == 'minmax':
        min_val = column_data.min()
        max_val = column_data.max()
        if max_val != min_val:
            normalized_df[column_name] = (column_data - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean = column_data.mean()
        std = column_data.std()
        if std != 0:
            normalized_df[column_name] = (column_data - mean) / std
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return normalized_df