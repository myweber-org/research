
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method."""
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
    """Normalize columns using min-max scaling."""
    df_norm = df.copy()
    for col in columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

def handle_missing_values(df, strategy='mean'):
    """Handle missing values with specified strategy."""
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
    
    return df_filled

def clean_dataframe(df, numeric_columns, outlier_removal=True, normalization=True, missing_strategy='mean'):
    """Main function to clean dataframe."""
    print(f"Original shape: {df.shape}")
    
    if outlier_removal:
        df = remove_outliers_iqr(df, numeric_columns)
        print(f"After outlier removal: {df.shape}")
    
    df = handle_missing_values(df, strategy=missing_strategy)
    print("Missing values handled")
    
    if normalization:
        df = normalize_minmax(df, numeric_columns)
        print("Data normalized")
    
    return df

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9],
        'feature2': [10, 20, 30, 40, 50, 200, 60, 70, 80, 90],
        'feature3': [0.1, 0.2, 0.3, 0.4, 0.5, 5.0, 0.6, 0.7, 0.8, 0.9]
    }
    
    df = pd.DataFrame(sample_data)
    numeric_cols = ['feature1', 'feature2', 'feature3']
    
    cleaned_df = clean_dataframe(
        df, 
        numeric_columns=numeric_cols,
        outlier_removal=True,
        normalization=True,
        missing_strategy='mean'
    )
    
    print("\nCleaned DataFrame:")
    print(cleaned_df.head())
    print(f"\nFinal shape: {cleaned_df.shape}")