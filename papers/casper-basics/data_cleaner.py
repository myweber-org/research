import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    """
    Load a CSV file and perform basic data cleaning operations.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Remove duplicate rows
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows - df.shape[0]
    if removed_duplicates > 0:
        print(f"Removed {removed_duplicates} duplicate rows.")

    # Handle missing values for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"Filled missing values in '{col}' with median.")

    # Remove outliers using Z-score for numeric columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    outlier_mask = (z_scores < 3).all(axis=1)
    df_clean = df[outlier_mask].copy()
    removed_outliers = df.shape[0] - df_clean.shape[0]
    if removed_outliers > 0:
        print(f"Removed {removed_outliers} rows containing outliers.")

    # Normalize numeric columns (Min-Max scaling)
    for col in numeric_cols:
        min_val = df_clean[col].min()
        max_val = df_clean[col].max()
        if max_val > min_val:
            df_clean[col] = (df_clean[col] - min_val) / (max_val - min_val)
            print(f"Normalized column '{col}' using Min-Max scaling.")

    print(f"Final cleaned data shape: {df_clean.shape}")
    return df_clean

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.
    """
    if df is not None and not df.empty:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return True
    else:
        print("No data to save.")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"

    cleaned_df = load_and_clean_data(input_file)
    if cleaned_df is not None:
        save_cleaned_data(cleaned_df, output_file)