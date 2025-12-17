import pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'")
    return filtered_df

def normalize_column(df, column):
    """Normalize column using min-max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val != min_val:
        df[column] = (df[column] - min_val) / (max_val - min_val)
        print(f"Normalized column '{column}' to range [0, 1]")
    else:
        print(f"Warning: Column '{column}' has constant values")
    return df

def clean_data(df, numeric_columns):
    """Main data cleaning pipeline."""
    if df is None:
        return None
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    for column in numeric_columns:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
    
    for column in numeric_columns:
        if column in df.columns:
            df = normalize_column(df, column)
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Removed {original_shape[0] - df.shape[0]} rows total")
    
    return df

def save_cleaned_data(df, output_path):
    """Save cleaned data to CSV."""
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return True
    return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ["age", "income", "score"]
    
    raw_data = load_data(input_file)
    if raw_data is not None:
        cleaned_data = clean_data(raw_data, numeric_cols)
        save_cleaned_data(cleaned_data, output_file)