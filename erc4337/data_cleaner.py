
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")

def normalize_data(df, columns):
    result = df.copy()
    for col in columns:
        if col in df.columns:
            result[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return result

def process_dataset(df, numeric_columns):
    df_clean = clean_missing_values(df, strategy='median')
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean = remove_outliers_iqr(df_clean, col)
    return normalize_data(df_clean, numeric_columns)