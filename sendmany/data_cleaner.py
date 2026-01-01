
import pandas as pd

def clean_dataset(df, subset=None, fill_method='mean'):
    """
    Clean a pandas DataFrame by removing duplicate rows and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    subset (list, optional): List of column names to consider for duplicate removal.
    fill_method (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates(subset=subset, keep='first')
    
    # Handle missing values
    if fill_method == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif fill_method == 'mean':
        df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
    elif fill_method == 'median':
        df_cleaned = df_cleaned.fillna(df_cleaned.median(numeric_only=True))
    elif fill_method == 'mode':
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else None)
            else:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 0)
    else:
        raise ValueError("fill_method must be 'drop', 'mean', 'median', or 'mode'")
    
    return df_cleaned

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        if df[numeric_cols].isin([float('inf'), float('-inf')]).any().any():
            print("DataFrame contains infinite values in numeric columns.")
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, subset=['A', 'C'], fill_method='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid = validate_data(cleaned_df, required_columns=['A', 'B', 'C'])
    print(f"\nData validation passed: {is_valid}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, method='iqr', normalize=True):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    if normalize:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = normalize_minmax(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def generate_summary(original_df, cleaned_df, numeric_columns):
    summary = {}
    for col in numeric_columns:
        if col in original_df.columns and col in cleaned_df.columns:
            summary[col] = {
                'original_count': len(original_df),
                'cleaned_count': len(cleaned_df),
                'removed_count': len(original_df) - len(cleaned_df),
                'original_mean': original_df[col].mean(),
                'cleaned_mean': cleaned_df[col].mean(),
                'original_std': original_df[col].std(),
                'cleaned_std': cleaned_df[col].std()
            }
    return pd.DataFrame(summary).T

if __name__ == "__main__":
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000)
    })
    
    sample_data.loc[np.random.choice(1000, 50), 'feature_a'] = 500
    
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    
    cleaned_data = clean_dataset(sample_data, numeric_cols, method='iqr', normalize=True)
    
    summary_stats = generate_summary(sample_data, cleaned_data, numeric_cols)
    
    print("Data cleaning completed successfully.")
    print(f"Original dataset shape: {sample_data.shape}")
    print(f"Cleaned dataset shape: {cleaned_data.shape}")
    print(f"Removed {len(sample_data) - len(cleaned_data)} outliers")
    print("\nSummary statistics:")
    print(summary_stats)