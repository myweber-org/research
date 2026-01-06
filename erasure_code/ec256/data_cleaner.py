import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    if columns is None:
        columns = df.columns
    return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    df_filled = df.copy()
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            df_filled[col] = df[col].fillna(df[col].mean())
    return df_filled

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    df_clean = df.copy()
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def standardize_columns(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    df_standardized = df.copy()
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df_standardized[col] = (df[col] - mean) / std
    return df_standardized

def clean_dataset(df, missing_strategy='remove', outlier_removal=True, standardization=True):
    df_clean = df.copy()
    
    if missing_strategy == 'remove':
        df_clean = remove_missing_rows(df_clean)
    elif missing_strategy == 'mean':
        df_clean = fill_missing_with_mean(df_clean)
    
    if outlier_removal:
        df_clean = remove_outliers_iqr(df_clean)
    
    if standardization:
        df_clean = standardize_columns(df_clean)
    
    return df_clean