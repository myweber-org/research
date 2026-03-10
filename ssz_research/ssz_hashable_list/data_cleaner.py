
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
    
    df_copy = dataframe.copy()
    
    if method == 'zscore':
        df_copy[f'{column}_normalized'] = stats.zscore(df_copy[column])
    
    elif method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
        else:
            df_copy[f'{column}_normalized'] = 0
    
    elif method == 'robust':
        median_val = df_copy[column].median()
        iqr_val = df_copy[column].quantile(0.75) - df_copy[column].quantile(0.25)
        if iqr_val != 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - median_val) / iqr_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df_copy

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5, normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric columns to process
    outlier_threshold (float): IQR threshold for outlier removal
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
    bool: True if validation passes, False otherwise
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False
    
    if len(dataframe) < min_rows:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False
    
    return True

def get_data_summary(dataframe):
    """
    Generate statistical summary of DataFrame.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing data summary
    """
    summary = {
        'total_rows': len(dataframe),
        'total_columns': len(dataframe.columns),
        'numeric_columns': dataframe.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': dataframe.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'memory_usage': dataframe.memory_usage(deep=True).sum()
    }
    
    return summary