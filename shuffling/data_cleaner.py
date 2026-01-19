import pandas as pd
import numpy as np
from scipy import stats

def clean_dataset(df, numeric_columns=None, z_threshold=3):
    """
    Clean dataset by handling missing values, normalizing numeric columns,
    and removing outliers based on z-score.
    """
    df_clean = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Handle missing values
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Normalize numeric columns
    for col in numeric_columns:
        if col in df_clean.columns:
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            if std_val > 0:
                df_clean[col] = (df_clean[col] - mean_val) / std_val
    
    # Remove outliers using z-score
    z_scores = np.abs(stats.zscore(df_clean[numeric_columns]))
    outlier_mask = (z_scores < z_threshold).all(axis=1)
    df_clean = df_clean[outlier_mask].reset_index(drop=True)
    
    return df_clean

def validate_data(df, required_columns):
    """
    Validate that required columns exist and have no null values.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found in columns: {null_counts[null_counts > 0].to_dict()}")
    
    return True

def sample_data(df, sample_size=1000, random_state=42):
    """
    Sample data from dataframe while maintaining distribution.
    """
    if len(df) <= sample_size:
        return df
    
    return df.sample(n=sample_size, random_state=random_state)