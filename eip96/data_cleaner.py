import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val == min_val:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def main():
    sample_data = {
        'feature_a': [10, 12, 13, 100, 11, 14, 9, 15, 200, 12],
        'feature_b': [1.2, 1.3, 1.1, 50.0, 1.4, 1.2, 1.3, 1.1, 1.5, 1.2],
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C']
    }
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    
    numeric_cols = ['feature_a', 'feature_b']
    cleaned = clean_dataset(df, numeric_cols)
    
    print("\nCleaned dataset:")
    print(cleaned)
    print(f"\nOriginal shape: {df.shape}, Cleaned shape: {cleaned.shape}")

if __name__ == "__main__":
    main()