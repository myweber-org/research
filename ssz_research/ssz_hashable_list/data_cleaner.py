
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
    df_norm = df.copy()
    for col in columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

def clean_dataset(input_path, output_path, numeric_cols):
    df = pd.read_csv(input_path)
    df_clean = remove_outliers_iqr(df, numeric_cols)
    df_normalized = normalize_minmax(df_clean, numeric_cols)
    df_normalized.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Original rows: {len(df)}, Cleaned rows: {len(df_normalized)}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_columns = ["age", "income", "score"]
    clean_dataset(input_file, output_file, numeric_columns)