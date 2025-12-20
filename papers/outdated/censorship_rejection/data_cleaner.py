
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns, method='minmax'):
    df_norm = df.copy()
    for col in columns:
        if method == 'minmax':
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        elif method == 'zscore':
            df_norm[col] = (df_norm[col] - df_norm[col].mean()) / df_norm[col].std()
        elif method == 'robust':
            df_norm[col] = (df_norm[col] - df_norm[col].median()) / stats.iqr(df_norm[col])
    return df_norm

def handle_missing_values(df, strategy='mean'):
    df_filled = df.copy()
    for col in df_filled.columns:
        if df_filled[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_filled[col].mean()
            elif strategy == 'median':
                fill_value = df_filled[col].median()
            elif strategy == 'mode':
                fill_value = df_filled[col].mode()[0]
            elif strategy == 'ffill':
                df_filled[col].fillna(method='ffill', inplace=True)
                continue
            elif strategy == 'bfill':
                df_filled[col].fillna(method='bfill', inplace=True)
                continue
            df_filled[col].fillna(fill_value, inplace=True)
    return df_filled

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    df_processed = df.copy()
    
    if outlier_method == 'iqr':
        df_processed = remove_outliers_iqr(df_processed, numeric_columns)
    
    df_processed = handle_missing_values(df_processed, strategy=missing_strategy)
    df_processed = normalize_data(df_processed, numeric_columns, method=normalize_method)
    
    return df_processed

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature3': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    }
    
    df = pd.DataFrame(sample_data)
    numeric_cols = ['feature1', 'feature2', 'feature3']
    
    cleaned_df = clean_dataset(df, numeric_cols)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)