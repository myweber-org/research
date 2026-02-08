
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_minmax(df, columns):
    """
    Normalize specified columns using Min-Max scaling.
    """
    df_norm = df.copy()
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0
    return df_norm

def clean_dataset(df, numeric_columns):
    """
    Main cleaning pipeline: remove outliers and normalize numeric columns.
    """
    if df.empty:
        return df
    
    df_clean = remove_outliers_iqr(df, numeric_columns)
    df_clean = normalize_minmax(df_clean, numeric_columns)
    df_clean = df_clean.reset_index(drop=True)
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'feature_a': np.random.normal(50, 15, 100),
        'feature_b': np.random.exponential(10, 100),
        'category': np.random.choice(['X', 'Y', 'Z'], 100)
    }
    df_sample = pd.DataFrame(sample_data)
    numeric_cols = ['feature_a', 'feature_b']
    
    cleaned_df = clean_dataset(df_sample, numeric_cols)
    print(f"Original shape: {df_sample.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(cleaned_df.head())