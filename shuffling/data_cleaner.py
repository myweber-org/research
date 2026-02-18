
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    elif missing_strategy == 'median':
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    elif missing_strategy == 'mode':
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    elif missing_strategy == 'drop':
        df_clean = df_clean.dropna()
    
    # Remove outliers using Z-score method
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((df_clean[numeric_cols] - df_clean[numeric_cols].mean()) / df_clean[numeric_cols].std())
    outlier_mask = (z_scores < outlier_threshold).all(axis=1)
    df_clean = df_clean[outlier_mask]
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

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
    
    if len(df) < min_rows:
        return False, f"Dataframe has less than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataframe is valid"

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    subset (list): Columns to consider for duplicates
    keep (str): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: Dataframe without duplicates
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): Columns to normalize
    method (str): Normalization method ('minmax', 'zscore')
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    df_norm = df.copy()
    
    if columns is None:
        columns = df_norm.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_norm.columns and pd.api.types.is_numeric_dtype(df_norm[col]):
            if method == 'minmax':
                col_min = df_norm[col].min()
                col_max = df_norm[col].max()
                if col_max != col_min:
                    df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
            elif method == 'zscore':
                col_mean = df_norm[col].mean()
                col_std = df_norm[col].std()
                if col_std != 0:
                    df_norm[col] = (df_norm[col] - col_mean) / col_std
    
    return df_norm

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Clean the data
    df_clean = clean_dataset(df, missing_strategy='mean', outlier_threshold=2)
    print("Cleaned DataFrame:")
    print(df_clean)
    print()
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(df_clean, required_columns=['A', 'B', 'C'])
    print(f"Validation: {is_valid} - {message}")
    print()
    
    # Remove duplicates
    df_no_dups = remove_duplicates(df_clean)
    print("DataFrame without duplicates:")
    print(df_no_dups)
    print()
    
    # Normalize columns
    df_normalized = normalize_columns(df_no_dups, method='minmax')
    print("Normalized DataFrame:")
    print(df_normalized)