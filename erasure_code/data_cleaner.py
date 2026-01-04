
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def zscore_normalize(df, column):
    """Normalize column using z-score normalization."""
    df[column] = stats.zscore(df[column])
    return df

def minmax_normalize(df, column):
    """Normalize column using min-max scaling."""
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def clean_data(df, numeric_columns, normalization_method='zscore'):
    """Main cleaning pipeline."""
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            
            if normalization_method == 'zscore':
                cleaned_df = zscore_normalize(cleaned_df, col)
            elif normalization_method == 'minmax':
                cleaned_df = minmax_normalize(cleaned_df, col)
    
    return cleaned_df

def save_cleaned_data(df, output_path):
    """Save cleaned dataframe to CSV."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ["feature1", "feature2", "feature3"]
    
    raw_data = load_dataset(input_file)
    cleaned_data = clean_data(raw_data, numeric_cols, normalization_method='zscore')
    save_cleaned_data(cleaned_data, output_file)
    
    print(f"Data cleaning completed. Original shape: {raw_data.shape}, Cleaned shape: {cleaned_data.shape}")