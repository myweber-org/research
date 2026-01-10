
import pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns=None, method='zscore'):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    if method == 'zscore':
        for col in columns:
            if col in df.columns:
                df_normalized[col] = stats.zscore(df[col])
    elif method == 'minmax':
        for col in columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    elif method == 'robust':
        for col in columns:
            if col in df.columns:
                median = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                if iqr > 0:
                    df_normalized[col] = (df[col] - median) / iqr
    
    return df_normalized

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    if method == 'iqr':
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                df_clean = df_clean[mask]
    
    elif method == 'zscore':
        for col in columns:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                mask = z_scores < threshold
                df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def handle_missing_values(df, columns=None, strategy='mean'):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'ffill':
                df_filled[col] = df[col].fillna(method='ffill')
                continue
            elif strategy == 'bfill':
                df_filled[col] = df[col].fillna(method='bfill')
                continue
            else:
                fill_value = 0
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def clean_dataset(df, numeric_columns=None, normalize=True, remove_outliers_flag=True, handle_missing=True):
    df_processed = df.copy()
    
    if handle_missing:
        df_processed = handle_missing_values(df_processed, numeric_columns)
    
    if remove_outliers_flag and numeric_columns:
        df_processed = remove_outliers(df_processed, numeric_columns)
    
    if normalize and numeric_columns:
        df_processed = normalize_data(df_processed, numeric_columns)
    
    return df_processed

def validate_data(df, check_duplicates=True, check_types=True):
    validation_report = {}
    
    validation_report['rows'] = len(df)
    validation_report['columns'] = len(df.columns)
    validation_report['missing_values'] = df.isnull().sum().sum()
    
    if check_duplicates:
        validation_report['duplicate_rows'] = df.duplicated().sum()
    
    if check_types:
        type_counts = df.dtypes.value_counts().to_dict()
        validation_report['data_types'] = {str(k): int(v) for k, v in type_counts.items()}
    
    return validation_report