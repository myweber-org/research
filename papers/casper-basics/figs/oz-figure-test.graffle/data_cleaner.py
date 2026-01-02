import numpy as np
import pandas as pd

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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.5)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    required_checks = [
        (lambda x: isinstance(x, pd.DataFrame), "Input must be a pandas DataFrame"),
        (lambda x: not x.empty, "DataFrame cannot be empty"),
        (lambda x: x.isnull().sum().sum() == 0, "DataFrame contains null values")
    ]
    for check_func, error_msg in required_checks:
        if not check_func(df):
            raise ValueError(error_msg)
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 50),
        'feature_b': np.random.exponential(2, 50),
        'category': np.random.choice(['X', 'Y', 'Z'], 50)
    })
    print("Original shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    print("Cleaned shape:", cleaned.shape)
    print("Data validation passed:", validate_dataframe(cleaned))import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers from a pandas Series using the IQR method.
    
    Parameters:
    data (pd.Series): Input data series
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.Series: Data with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(series >= lower_bound) & (series <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized data series
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column]
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    
    return (series - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization (mean=0, std=1).
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.Series: Standardized data series
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column]
    mean_val = series.mean()
    std_val = series.std()
    
    if std_val == 0:
        return pd.Series([0] * len(series), index=series.index)
    
    return (series - mean_val) / std_val

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='standardize'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    outlier_method (str): 'iqr' or None for outlier removal
    normalize_method (str): 'minmax', 'standardize', or None for normalization
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if df.empty:
        return df
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'standardize':
            cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"