
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    mask = z_scores < threshold
    return data[mask]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame or Series
        column: column name to normalize
    
    Returns:
        Normalized data
    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column]
    else:
        series = data
    
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return series * 0  # Return zeros if all values are same
    
    return (series - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame or Series
        column: column name to normalize
    
    Returns:
        Standardized data
    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column]
    else:
        series = data
    
    mean_val = series.mean()
    std_val = series.std()
    
    if std_val == 0:
        return series * 0  # Return zeros if no variance
    
    return (series - mean_val) / std_val

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric columns to process (default: all numeric)
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    # Remove outliers
    if outlier_method:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                if outlier_method == 'iqr':
                    cleaned_df = remove_outliers_iqr(cleaned_df, col)
                elif outlier_method == 'zscore':
                    cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    # Normalize data
    if normalize_method:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                if normalize_method == 'minmax':
                    cleaned_df[col] = normalize_minmax(cleaned_df, col)
                elif normalize_method == 'zscore':
                    cleaned_df[col] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns=None, check_missing=True, check_duplicates=True):
    """
    Validate dataset for common issues.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required columns
        check_missing: Check for missing values
        check_duplicates: Check for duplicate rows
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_values': {},
        'duplicate_count': 0,
        'missing_columns': []
    }
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['missing_columns'] = missing_cols
            validation_results['is_valid'] = False
    
    # Check missing values
    if check_missing:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            validation_results['missing_values'] = missing.to_dict()
            validation_results['is_valid'] = False
    
    # Check duplicates
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results['duplicate_count'] = duplicate_count
            validation_results['is_valid'] = False
    
    return validation_resultsimport pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for required columns and minimum row count.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (bool, str) indicating validation result and error message.
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid} - {message}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
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
    Standardize data using z-score normalization
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
    Handle missing values in numeric columns
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
    elif strategy == 'mode':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
    elif strategy == 'drop':
        data = data.dropna(subset=numeric_cols)
    
    return data

def create_cleaning_pipeline(data, config):
    """
    Apply multiple cleaning operations based on configuration
    """
    cleaned_data = data.copy()
    
    for operation in config.get('operations', []):
        op_type = operation.get('type')
        
        if op_type == 'remove_outliers':
            cleaned_data = remove_outliers_iqr(
                cleaned_data, 
                operation['column'], 
                operation.get('threshold', 1.5)
            )
        elif op_type == 'normalize':
            col = operation['column']
            method = operation.get('method', 'minmax')
            
            if method == 'minmax':
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
            elif method == 'zscore':
                cleaned_data[col] = standardize_zscore(cleaned_data, col)
        elif op_type == 'handle_missing':
            cleaned_data = handle_missing_values(
                cleaned_data, 
                operation.get('strategy', 'mean')
            )
    
    return cleaned_data
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10],
        'humidity': [45, 46, 47, 48, 49, 200, 50, 51, 52, -5],
        'pressure': [1013, 1014, 1015, 1016, 1017, 2000, 1018, 1019, 1020, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df = clean_numeric_data(df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    print("\nCleaned DataFrame:")
    print(cleaned_df)