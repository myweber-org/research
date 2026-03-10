
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    return filtered_df

def remove_outliers_zscore(dataframe, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(dataframe[column]))
    filtered_df = dataframe[z_scores < threshold]
    return filtered_df

def normalize_minmax(dataframe, columns):
    """
    Normalize specified columns using Min-Max scaling
    """
    df_normalized = dataframe.copy()
    for col in columns:
        if col in df_normalized.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val != min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def normalize_zscore(dataframe, columns):
    """
    Normalize specified columns using Z-score standardization
    """
    df_normalized = dataframe.copy()
    for col in columns:
        if col in df_normalized.columns:
            mean_val = df_normalized[col].mean()
            std_val = df_normalized[col].std()
            if std_val > 0:
                df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    return df_normalized

def clean_dataset(dataframe, outlier_method='iqr', normalize_method='minmax', 
                  numeric_columns=None, outlier_factor=1.5, zscore_threshold=3):
    """
    Main cleaning function that applies outlier removal and normalization
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            if outlier_method == 'iqr':
                cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_factor)
            elif outlier_method == 'zscore':
                cleaned_df = remove_outliers_zscore(cleaned_df, column, zscore_threshold)
    
    if normalize_method == 'minmax':
        cleaned_df = normalize_minmax(cleaned_df, numeric_columns)
    elif normalize_method == 'zscore':
        cleaned_df = normalize_zscore(cleaned_df, numeric_columns)
    
    return cleaned_df

def get_cleaning_summary(original_df, cleaned_df):
    """
    Generate summary statistics comparing original and cleaned data
    """
    summary = {}
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in original_df.columns and col in cleaned_df.columns:
            summary[col] = {
                'original_rows': len(original_df),
                'cleaned_rows': len(cleaned_df),
                'original_mean': original_df[col].mean(),
                'cleaned_mean': cleaned_df[col].mean(),
                'original_std': original_df[col].std(),
                'cleaned_std': cleaned_df[col].std(),
                'removed_outliers': len(original_df) - len(cleaned_df)
            }
    
    return pd.DataFrame(summary).T