
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_column(dataframe, column, method='zscore'):
    """
    Normalize a column using specified method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('zscore', 'minmax', 'robust')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result_df = dataframe.copy()
    
    if method == 'zscore':
        result_df[f'{column}_normalized'] = stats.zscore(result_df[column])
    
    elif method == 'minmax':
        col_min = result_df[column].min()
        col_max = result_df[column].max()
        if col_max != col_min:
            result_df[f'{column}_normalized'] = (result_df[column] - col_min) / (col_max - col_min)
        else:
            result_df[f'{column}_normalized'] = 0.5
    
    elif method == 'robust':
        median = result_df[column].median()
        iqr = result_df[column].quantile(0.75) - result_df[column].quantile(0.25)
        if iqr != 0:
            result_df[f'{column}_normalized'] = (result_df[column] - median) / iqr
        else:
            result_df[f'{column}_normalized'] = 0
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return result_df

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5, normalize_method='zscore'):
    """
    Comprehensive dataset cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric columns to process
    outlier_threshold (float): IQR multiplier for outlier removal
    normalize_method (str): Normalization method to apply
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_threshold)
            cleaned_df = normalize_column(cleaned_df, column, normalize_method)
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    dataframe (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame validation passed"