import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fillna_strategy='mean', columns=None):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fillna_strategy (str): Strategy for filling NaN values.
                               Options: 'mean', 'median', 'mode', 'zero', 'ffill'.
        columns (list): Specific columns to apply cleaning. If None, applies to all.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns.tolist()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    for col in columns:
        if col in df_clean.columns:
            if df_clean[col].dtype in [np.float64, np.int64]:
                if fillna_strategy == 'mean':
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif fillna_strategy == 'median':
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif fillna_strategy == 'zero':
                    df_clean[col].fillna(0, inplace=True)
                elif fillna_strategy == 'ffill':
                    df_clean[col].fillna(method='ffill', inplace=True)
            elif df_clean[col].dtype == object:
                if fillna_strategy == 'mode':
                    mode_val = df_clean[col].mode()
                    df_clean[col].fillna(mode_val[0] if not mode_val.empty else '', inplace=True)
                elif fillna_strategy == 'ffill':
                    df_clean[col].fillna(method='ffill', inplace=True)
                else:
                    df_clean[col].fillna('', inplace=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): Columns to check for outliers. If None, uses all numeric columns.
        multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    mask = pd.Series([True] * len(df_clean))
    
    for col in columns:
        if col in df_clean.columns and df_clean[col].dtype in [np.float64, np.int64]:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            col_mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            mask = mask & col_mask
    
    return df_clean[mask].reset_index(drop=True)