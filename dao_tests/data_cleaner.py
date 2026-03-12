
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    df = df.dropna()
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv', ['age', 'income', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Cleaned data shape: {cleaned_data.shape}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling to range [0, 1].
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names (if None, auto-detect)
        outlier_multiplier: IQR multiplier for outlier removal
    
    Returns:
        Tuple of (cleaned_data, normalization_report)
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    report = {}
    
    for col in numeric_columns:
        if col not in data.columns:
            continue
            
        original_count = len(cleaned_data)
        cleaned_data = remove_outliers_iqr(cleaned_data, col, outlier_multiplier)
        outliers_removed = original_count - len(cleaned_data)
        
        cleaned_data[col + '_normalized'] = normalize_minmax(cleaned_data, col)
        cleaned_data[col + '_standardized'] = standardize_zscore(cleaned_data, col)
        
        report[col] = {
            'outliers_removed': outliers_removed,
            'original_mean': data[col].mean(),
            'cleaned_mean': cleaned_data[col].mean(),
            'original_std': data[col].std(),
            'cleaned_std': cleaned_data[col].std()
        }
    
    return cleaned_data, report

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        allow_nan: whether NaN values are allowed
    
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(data, pd.DataFrame):
        validation_result['is_valid'] = False
        validation_result['errors'].append('Input is not a pandas DataFrame')
        return validation_result
    
    if len(data) == 0:
        validation_result['is_valid'] = False
        validation_result['errors'].append('DataFrame is empty')
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Missing required columns: {missing_columns}')
    
    if not allow_nan:
        nan_columns = data.columns[data.isna().any()].tolist()
        if nan_columns:
            validation_result['warnings'].append(f'Columns with NaN values: {nan_columns}')
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].std() == 0:
            validation_result['warnings'].append(f'Column {col} has zero variance')
    
    return validation_result