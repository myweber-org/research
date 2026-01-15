
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Remove duplicate rows and standardize column names.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')
    
    return df_cleaned

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'drop':
        df = df.dropna(subset=numeric_cols)
    
    return df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
        'Age': [25, 30, 25, 35, 40],
        'Score': [85.5, 90.0, 85.5, None, 95.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    df_cleaned = clean_dataset(df)
    print("\nAfter cleaning duplicates and standardizing columns:")
    print(df_cleaned)
    
    df_filled = handle_missing_values(df_cleaned, strategy='mean')
    print("\nAfter handling missing values:")
    print(df_filled)
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: If True, fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if fill_missing:
        df_clean = df_clean.fillna(fill_value)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers from a DataFrame column using z-score method.
    
    Args:
        df: pandas DataFrame
        column: Column name to process
        threshold: Z-score threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    from scipy import stats
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = stats.zscore(df[column].dropna())
    abs_z_scores = abs(z_scores)
    
    filtered_entries = (abs_z_scores < threshold)
    return df[filtered_entries]
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, columns_to_clean):
    cleaned_df = df.copy()
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    return cleaned_df.reset_index(drop=True)