
import pandas as pd
import numpy as np

def remove_outliers(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return df[z_scores < threshold]

def normalize_column(df, column, method='minmax'):
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        if std != 0:
            df[column] = (df[column] - mean) / std
    return df

def clean_dataframe(df, numeric_columns, outlier_threshold=3, normalize_method='minmax'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers(cleaned_df, col, outlier_threshold)
            cleaned_df = normalize_column(cleaned_df, col, normalize_method)
    
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def main():
    sample_data = {
        'A': [1, 2, 3, 100, 5, 6, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df, ['A', 'B', 'C'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    main()