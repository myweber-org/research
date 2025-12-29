
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method for specified columns."""
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns, method='minmax'):
    """Normalize specified columns using min-max or z-score normalization."""
    df_norm = df.copy()
    for col in columns:
        if col in df.columns:
            if method == 'minmax':
                df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            elif method == 'zscore':
                df_norm[col] = (df[col] - df[col].mean()) / df[col].std()
    return df_norm

def handle_missing_values(df, strategy='mean'):
    """Handle missing values using specified strategy."""
    df_filled = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_filled[col] = df[col].fillna(df[col].mode()[0])
    
    return df_filled

def clean_data_pipeline(input_file, output_file, numeric_columns):
    """Complete data cleaning pipeline."""
    print(f"Loading data from {input_file}")
    df = load_dataset(input_file)
    
    print(f"Original shape: {df.shape}")
    
    print("Handling missing values...")
    df = handle_missing_values(df, strategy='median')
    
    print("Removing outliers...")
    df = remove_outliers_iqr(df, numeric_columns)
    print(f"Shape after outlier removal: {df.shape}")
    
    print("Normalizing data...")
    df = normalize_data(df, numeric_columns, method='minmax')
    
    print(f"Saving cleaned data to {output_file}")
    df.to_csv(output_file, index=False)
    
    return df

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    numeric_cols = ['age', 'income', 'score']
    
    cleaned_df = clean_data_pipeline(input_csv, output_csv, numeric_cols)
    print("Data cleaning completed successfully.")