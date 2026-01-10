import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using IQR method.
    """
    clean_df = df.copy()
    for col in columns:
        if col in clean_df.columns:
            Q1 = clean_df[col].quantile(0.25)
            Q3 = clean_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
    return clean_df

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    clean_df = df.copy()
    for col in columns:
        if col in clean_df.columns:
            z_scores = np.abs(stats.zscore(clean_df[col]))
            clean_df = clean_df[z_scores < threshold]
    return clean_df

def normalize_minmax(df, columns):
    """
    Normalize data using Min-Max scaling.
    """
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def normalize_zscore(df, columns):
    """
    Normalize data using Z-score standardization.
    """
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            if std_val != 0:
                normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    return normalized_df

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing.
    """
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df, numeric_columns)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df, numeric_columns)
    else:
        df_clean = df.copy()
    
    if normalize_method == 'minmax':
        df_final = normalize_minmax(df_clean, numeric_columns)
    elif normalize_method == 'zscore':
        df_final = normalize_zscore(df_clean, numeric_columns)
    else:
        df_final = df_clean
    
    return df_final

if __name__ == "__main__":
    sample_data = {
        'feature1': [10, 12, 12, 13, 12, 50, 11, 14, 13, 12],
        'feature2': [100, 120, 115, 118, 122, 500, 112, 119, 117, 121],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df_sample = pd.DataFrame(sample_data)
    numeric_cols = ['feature1', 'feature2']
    
    print("Original dataset:")
    print(df_sample)
    print("\nCleaned dataset (IQR + MinMax):")
    cleaned_df = clean_dataset(df_sample, numeric_cols, outlier_method='iqr', normalize_method='minmax')
    print(cleaned_df)