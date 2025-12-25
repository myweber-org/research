
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
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

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if not np.issubdtype(normalized_df[col].dtype, np.number):
            raise TypeError(f"Column '{col}' must be numeric")
        
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        
        if col_max != col_min:
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        else:
            normalized_df[col] = 0
    
    return normalized_df

def detect_skewed_columns(dataframe, skew_threshold=0.5):
    """
    Detect columns with skewed distributions.
    
    Args:
        dataframe: pandas DataFrame
        skew_threshold: absolute skewness threshold (default 0.5)
    
    Returns:
        Dictionary with skewness values for columns exceeding threshold
    """
    skewed_cols = {}
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        skewness = stats.skew(dataframe[col].dropna())
        if abs(skewness) > skew_threshold:
            skewed_cols[col] = skewness
    
    return skewed_cols

def clean_dataset(dataframe, outlier_columns=None, normalize=True, skew_threshold=0.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame to clean
        outlier_columns: list of columns for outlier removal (default: all numeric)
        normalize: whether to normalize numeric columns
        skew_threshold: skewness detection threshold
    
    Returns:
        Tuple of (cleaned DataFrame, cleaning statistics)
    """
    df_cleaned = dataframe.copy()
    stats_dict = {
        'original_shape': dataframe.shape,
        'outliers_removed': 0,
        'skewed_columns': {}
    }
    
    if outlier_columns is None:
        outlier_columns = list(dataframe.select_dtypes(include=[np.number]).columns)
    
    for col in outlier_columns:
        if col in df_cleaned.columns:
            original_len = len(df_cleaned)
            df_cleaned = remove_outliers_iqr(df_cleaned, col)
            stats_dict['outliers_removed'] += original_len - len(df_cleaned)
    
    if normalize:
        df_cleaned = normalize_minmax(df_cleaned)
        stats_dict['normalized'] = True
    else:
        stats_dict['normalized'] = False
    
    skewed_cols = detect_skewed_columns(df_cleaned, skew_threshold)
    stats_dict['skewed_columns'] = skewed_cols
    stats_dict['final_shape'] = df_cleaned.shape
    
    return df_cleaned, stats_dict

def save_cleaning_report(stats_dict, output_path='cleaning_report.txt'):
    """
    Save cleaning statistics to a text file.
    
    Args:
        stats_dict: dictionary containing cleaning statistics
        output_path: path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("Data Cleaning Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Original dataset shape: {stats_dict['original_shape']}\n")
        f.write(f"Final dataset shape: {stats_dict['final_shape']}\n")
        f.write(f"Total outliers removed: {stats_dict['outliers_removed']}\n")
        f.write(f"Normalization applied: {stats_dict['normalized']}\n\n")
        
        if stats_dict['skewed_columns']:
            f.write("Skewed columns detected:\n")
            for col, skewness in stats_dict['skewed_columns'].items():
                f.write(f"  {col}: skewness = {skewness:.3f}\n")
        else:
            f.write("No significantly skewed columns detected.\n")