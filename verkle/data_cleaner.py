
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
            cleaned_df = standardize_zscore(cleaned_df, col)
    return cleaned_df

def generate_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000)
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = generate_sample_data()
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    cleaned_data = clean_dataset(sample_df, numeric_cols)
    print(f"Original shape: {sample_df.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(cleaned_data.describe())import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df, column_type_map):
    """
    Convert columns to specified data types.
    
    Args:
        df: pandas DataFrame
        column_type_map: dict mapping column names to target types
    
    Returns:
        DataFrame with converted columns
    """
    df_copy = df.copy()
    for column, dtype in column_type_map.items():
        if column in df_copy.columns:
            df_copy[column] = df_copy[column].astype(dtype)
    return df_copy

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(fill_value)
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def normalize_column(df, column):
    """
    Normalize a column to range [0, 1].
    
    Args:
        df: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        return df
    
    df_copy = df.copy()
    col_min = df_copy[column].min()
    col_max = df_copy[column].max()
    
    if col_max != col_min:
        df_copy[column] = (df_copy[column] - col_min) / (col_max - col_min)
    
    return df_copy

def clean_dataframe(df, operations):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df: pandas DataFrame
        operations: list of dicts, each specifying an operation and parameters
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for operation in operations:
        op_type = operation.get('type')
        params = operation.get('params', {})
        
        if op_type == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, **params)
        elif op_type == 'convert_types':
            cleaned_df = convert_column_types(cleaned_df, **params)
        elif op_type == 'handle_missing':
            cleaned_df = handle_missing_values(cleaned_df, **params)
        elif op_type == 'normalize':
            cleaned_df = normalize_column(cleaned_df, **params)
    
    return cleaned_dfimport numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        column: Column name to process
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from specified numeric columns.
    
    Args:
        df: pandas DataFrame
        numeric_columns: List of numeric column names to clean. If None, uses all numeric columns.
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        df: pandas DataFrame
        column: Column name
    
    Returns:
        Dictionary of statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'pressure': [1013, 1012, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1100]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nDataset statistics before cleaning:")
    for col in df.columns:
        stats = calculate_statistics(df, col)
        print(f"{col}: {stats}")
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned dataset:")
    print(cleaned_df)
    print(f"\nRemoved {len(df) - len(cleaned_df)} total outliers")