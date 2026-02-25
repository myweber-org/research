
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns boolean mask for outliers.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
    return outliers

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    Returns filtered DataFrame.
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column using Min-Max scaling.
    Returns normalized Series.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize column using Z-score normalization.
    Returns standardized Series.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='standardize'):
    """
    Main cleaning function that processes multiple numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        # Handle outliers
        if outlier_method == 'iqr':
            outliers = detect_outliers_iqr(cleaned_df, col)
            cleaned_df.loc[outliers, col] = np.nan
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        # Handle normalization
        if normalize_method == 'minmax':
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'standardize':
            cleaned_df[f'{col}_standardized'] = standardize_zscore(cleaned_df, col)
    
    # Fill NaN values with column mean
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
    
    return cleaned_df

def get_summary_statistics(df, numeric_columns):
    """
    Generate summary statistics for numeric columns.
    """
    summary = {}
    
    for col in numeric_columns:
        if col in df.columns:
            summary[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing': df[col].isna().sum(),
                'missing_percentage': (df[col].isna().sum() / len(df)) * 100
            }
    
    return pd.DataFrame(summary).T