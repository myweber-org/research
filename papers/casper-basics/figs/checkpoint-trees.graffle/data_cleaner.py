
import pandas as pd
import numpy as np

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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

def main():
    sample_data = {
        'feature_a': [10, 12, 14, 100, 15, 13, 11, 9, 16, 12],
        'feature_b': [20, 22, 24, 200, 25, 23, 21, 19, 26, 22],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    
    numeric_cols = ['feature_a', 'feature_b']
    cleaned_df = clean_dataset(df, numeric_cols)
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    print(f"\nOriginal shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")

if __name__ == "__main__":
    main()