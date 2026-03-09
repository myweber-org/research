
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    threshold (float): Z-score threshold (default 3)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    filtered_data = data.iloc[filtered_indices].copy()
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric columns to process
    outlier_method (str): 'iqr' or 'zscore' for outlier removal
    normalize_method (str): 'minmax' or 'zscore' for normalization
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Remove outliers
    for column in numeric_columns:
        if column in cleaned_data.columns:
            if outlier_method == 'iqr':
                cleaned_data = remove_outliers_iqr(cleaned_data, column)
            elif outlier_method == 'zscore':
                cleaned_data = remove_outliers_zscore(cleaned_data, column)
    
    # Normalize data
    for column in numeric_columns:
        if column in cleaned_data.columns:
            if normalize_method == 'minmax':
                cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
            elif normalize_method == 'zscore':
                cleaned_data[f'{column}_standardized'] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, check_missing=True, check_duplicates=True):
    """
    Validate dataset for common issues.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    required_columns (list): List of required columns
    check_missing (bool): Check for missing values
    check_duplicates (bool): Check for duplicate rows
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'missing_counts': {},
        'duplicate_count': 0
    }
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_cols}")
    
    # Check missing values
    if check_missing:
        missing_counts = data.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        if missing_counts:
            validation_results['missing_counts'] = missing_counts
            validation_results['issues'].append(f"Missing values found in columns: {list(missing_counts.keys())}")
    
    # Check duplicates
    if check_duplicates:
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            validation_results['duplicate_count'] = duplicate_count
            validation_results['issues'].append(f"Found {duplicate_count} duplicate rows")
    
    return validation_results
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using Interquartile Range method
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

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values using specified strategy
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            data[col].fillna(data[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            data[col].fillna(data[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)
    elif strategy == 'drop':
        data.dropna(inplace=True)
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'")
    
    return data

def detect_skewness(data, column, threshold=0.5):
    """
    Detect skewness in data column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    skewness = stats.skew(data[column].dropna())
    is_skewed = abs(skewness) > threshold
    
    return skewness, is_skewed

def log_transform(data, column):
    """
    Apply log transformation to reduce skewness
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if (data[column] <= 0).any():
        shifted_data = data[column] - data[column].min() + 1
        transformed = np.log(shifted_data)
    else:
        transformed = np.log(data[column])
    
    return transformed

def create_cleaning_pipeline(data, config):
    """
    Create a data cleaning pipeline based on configuration
    """
    cleaned_data = data.copy()
    report = {}
    
    for column in config.get('outlier_removal', []):
        if column in cleaned_data.columns:
            cleaned_data, removed = remove_outliers_iqr(cleaned_data, column)
            report[f'{column}_outliers_removed'] = removed
    
    for column in config.get('normalize', []):
        if column in cleaned_data.columns:
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
    
    for column in config.get('standardize', []):
        if column in cleaned_data.columns:
            cleaned_data[f'{column}_standardized'] = standardize_zscore(cleaned_data, column)
    
    if 'missing_values_strategy' in config:
        cleaned_data = handle_missing_values(cleaned_data, config['missing_values_strategy'])
    
    for column in config.get('transform_skewed', []):
        if column in cleaned_data.columns:
            skewness, is_skewed = detect_skewness(cleaned_data, column)
            if is_skewed:
                cleaned_data[f'{column}_log_transformed'] = log_transform(cleaned_data, column)
                report[f'{column}_skewness'] = skewness
    
    return cleaned_data, report

def validate_dataframe(data):
    """
    Validate DataFrame structure and content
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    validation_report = {
        'rows': len(data),
        'columns': len(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'duplicate_rows': data.duplicated().sum(),
        'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(data.select_dtypes(include=['object', 'category']).columns)
    }
    
    return validation_report