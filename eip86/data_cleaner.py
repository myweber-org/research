
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process, None for all numeric columns
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize, None for all numeric columns
    feature_range (tuple): Desired range of transformed data (default 0-1)
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max - col_min != 0:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
                df_normalized[col] = df_normalized[col] * (max_val - min_val) + min_val
    
    return df_normalized

def zscore_normalize(df, columns=None, threshold=3):
    """
    Normalize data using Z-score and optionally cap extreme values.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize, None for all numeric columns
    threshold (float): Z-score threshold for capping (default 3)
    
    Returns:
    pd.DataFrame: Dataframe with Z-score normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_zscore = df.copy()
    
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val != 0:
                z_scores = (df[col] - mean_val) / std_val
                df_zscore[col] = np.clip(z_scores, -threshold, threshold)
    
    return df_zscore

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): List of column names to process, None for all columns
    
    Returns:
    pd.DataFrame: Dataframe with handled missing values
    """
    if columns is None:
        columns = df.columns
    
    df_processed = df.copy()
    
    if strategy == 'drop':
        return df_processed.dropna(subset=columns).reset_index(drop=True)
    
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                df_processed[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                df_processed[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df_processed[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
    
    return df_processed

def create_data_quality_report(df):
    """
    Generate a data quality report for the dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Data quality report
    """
    report_data = []
    
    for col in df.columns:
        col_data = {
            'column': col,
            'dtype': str(df[col].dtype),
            'total_count': len(df[col]),
            'missing_count': df[col].isnull().sum(),
            'missing_percentage': (df[col].isnull().sum() / len(df[col])) * 100,
            'unique_count': df[col].nunique(),
            'unique_percentage': (df[col].nunique() / len(df[col])) * 100
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_data.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                '25%': df[col].quantile(0.25),
                'median': df[col].median(),
                '75%': df[col].quantile(0.75),
                'max': df[col].max()
            })
        
        report_data.append(col_data)
    
    return pd.DataFrame(report_data)