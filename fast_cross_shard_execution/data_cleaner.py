import pandas as pd
import numpy as np
from scipy import stats

def clean_dataset(df, numeric_columns=None, method='median', z_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    method (str): Imputation method ('mean', 'median', 'mode')
    z_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if df.empty:
        return df
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    # Handle missing values
    for col in numeric_columns:
        if col in df_clean.columns:
            if method == 'mean':
                fill_value = df_clean[col].mean()
            elif method == 'median':
                fill_value = df_clean[col].median()
            elif method == 'mode':
                fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
            else:
                fill_value = 0
            
            df_clean[col].fillna(fill_value, inplace=True)
    
    # Remove outliers using z-score
    z_scores = np.abs(stats.zscore(df_clean[numeric_columns]))
    outlier_mask = (z_scores < z_threshold).all(axis=1)
    df_clean = df_clean[outlier_mask].reset_index(drop=True)
    
    return df_clean

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): Columns to normalize
    method (str): Normalization method ('minmax', 'zscore')
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_norm = df.copy()
    
    for col in columns:
        if col in df_norm.columns and df_norm[col].dtype in [np.float64, np.int64]:
            if method == 'minmax':
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val > min_val:
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df_norm[col].mean()
                std_val = df_norm[col].std()
                if std_val > 0:
                    df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'feature_a': [1, 2, np.nan, 4, 100],
        'feature_b': [5, 6, 7, np.nan, 8],
        'feature_c': [10, 11, 12, 13, 14],
        'category': ['A', 'B', 'A', 'B', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50)
    
    # Clean the data
    cleaned_df = clean_dataset(df, numeric_columns=['feature_a', 'feature_b', 'feature_c'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50)
    
    # Normalize the data
    normalized_df = normalize_data(cleaned_df, method='minmax')
    print("Normalized DataFrame:")
    print(normalized_df)
    print("\n" + "="*50)
    
    # Validate the dataframe
    is_valid, message = validate_dataframe(normalized_df, min_rows=2)
    print(f"Validation: {is_valid}")
    print(f"Message: {message}")