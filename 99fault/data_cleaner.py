import pandas as pd
import numpy as np

def remove_missing_rows(df, threshold=0.5):
    """
    Remove rows with missing values exceeding the threshold percentage.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        threshold (float): Maximum allowed missing value ratio per row (0-1)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    missing_ratio = df.isnull().sum(axis=1) / df.shape[1]
    return df[missing_ratio <= threshold].reset_index(drop=True)

def replace_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Replace outliers with column boundaries using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to process
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers replaced
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        df_clean[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize specified columns to zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        if std_val > 0:
            df_standardized[col] = (df[col] - mean_val) / std_val
    
    return df_standardized

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Dictionary containing validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_result['is_valid'] = False
        validation_result['errors'].append('Input is not a pandas DataFrame')
        return validation_result
    
    if df.empty:
        validation_result['warnings'].append('DataFrame is empty')
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Missing required columns: {missing_columns}')
    
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        validation_result['warnings'].append(f'Found {duplicate_rows} duplicate rows')
    
    return validation_result