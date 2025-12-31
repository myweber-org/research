
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_data(df, columns, method='minmax'):
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            if method == 'minmax':
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                else:
                    normalized_df[col] = 0
            elif method == 'zscore':
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val != 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
                else:
                    normalized_df[col] = 0
    return normalized_df

def clean_dataset(file_path, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    try:
        df = pd.read_csv(file_path)
        if outlier_method == 'iqr':
            df_clean = remove_outliers_iqr(df, numeric_columns)
        else:
            df_clean = df.copy()
        
        df_normalized = normalize_data(df_clean, numeric_columns, normalize_method)
        return df_normalized
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'feature3': np.random.uniform(0, 1000, 200)
    })
    
    sample_data.to_csv('sample_dataset.csv', index=False)
    
    cleaned_data = clean_dataset(
        'sample_dataset.csv',
        ['feature1', 'feature2', 'feature3'],
        outlier_method='iqr',
        normalize_method='zscore'
    )
    
    if cleaned_data is not None:
        print(f"Original shape: {sample_data.shape}")
        print(f"Cleaned shape: {cleaned_data.shape}")
        print("\nCleaned data statistics:")
        print(cleaned_data.describe())
        cleaned_data.to_csv('cleaned_dataset.csv', index=False)