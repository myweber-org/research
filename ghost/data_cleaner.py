
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Clean missing values in a DataFrame using specified strategy.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns (list): Specific columns to clean, if None cleans all columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == 'mode':
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
        else:
            df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
    
    return df_clean

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a specific column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    method (str): Method for outlier detection ('iqr' or 'zscore')
    threshold (float): Threshold value for outlier detection
    
    Returns:
    pd.Series: Boolean series indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        return pd.Series([False] * len(df), index=df.index)
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val == 0:
            return pd.Series([False] * len(df), index=df.index)
        z_scores = np.abs((df[column] - mean_val) / std_val)
        return z_scores > threshold
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"