
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
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
    
    return filtered_df

def z_score_normalize(dataframe, columns=None):
    """
    Apply z-score normalization to specified columns.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            mean_val = dataframe[col].mean()
            std_val = dataframe[col].std()
            
            if std_val > 0:
                normalized_df[col] = (dataframe[col] - mean_val) / std_val
            else:
                normalized_df[col] = 0
    
    return normalized_df

def min_max_normalize(dataframe, columns=None, feature_range=(0, 1)):
    """
    Apply min-max normalization to specified columns.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of column names to normalize
        feature_range: Desired range of transformed data (default: (0, 1))
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    normalized_df = dataframe.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            
            if col_max > col_min:
                normalized_df[col] = ((dataframe[col] - col_min) / (col_max - col_min)) * (max_val - min_val) + min_val
            else:
                normalized_df[col] = min_val
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Args:
        dataframe: pandas DataFrame
        strategy: Imputation strategy ('mean', 'median', 'mode', 'constant', 'drop')
        columns: List of column names to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns.tolist()
    
    processed_df = dataframe.copy()
    
    for col in columns:
        if col not in processed_df.columns:
            continue
            
        if processed_df[col].isnull().any():
            if strategy == 'mean' and np.issubdtype(processed_df[col].dtype, np.number):
                processed_df[col].fillna(processed_df[col].mean(), inplace=True)
            elif strategy == 'median' and np.issubdtype(processed_df[col].dtype, np.number):
                processed_df[col].fillna(processed_df[col].median(), inplace=True)
            elif strategy == 'mode':
                processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
            elif strategy == 'constant':
                processed_df[col].fillna(0, inplace=True)
            elif strategy == 'drop':
                processed_df = processed_df.dropna(subset=[col])
    
    return processed_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"