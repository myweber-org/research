
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

def normalize_minmax(df, columns):
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        if max_val != min_val:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0
    return df_normalized

def clean_dataset(filepath, numeric_columns):
    try:
        df = pd.read_csv(filepath)
        print(f"Original shape: {df.shape}")
        
        df = remove_outliers_iqr(df, numeric_columns)
        print(f"After outlier removal: {df.shape}")
        
        df = normalize_minmax(df, numeric_columns)
        print("Data normalization completed")
        
        return df
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'feature_c': np.random.uniform(0, 1000, 200)
    })
    
    cleaned = clean_dataset('sample_data.csv', ['feature_a', 'feature_b', 'feature_c'])
    if cleaned is not None:
        cleaned.to_csv('cleaned_data.csv', index=False)
        print("Cleaned data saved to 'cleaned_data.csv'")