
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            continue
            
        if normalized_df[col].dtype in [np.float64, np.int64]:
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            
            if col_max != col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0.0
    
    return normalized_df

def clean_dataset(dataframe, numeric_columns=None, outlier_multiplier=1.5):
    """
    Complete data cleaning pipeline: remove outliers and normalize.
    
    Args:
        dataframe: pandas DataFrame
        numeric_columns: List of numeric columns to process
        outlier_multiplier: IQR multiplier for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_multiplier)
    
    cleaned_df = normalize_minmax(cleaned_df, numeric_columns)
    
    return cleaned_df.reset_index(drop=True)

def calculate_statistics(dataframe):
    """
    Calculate basic statistics for numeric columns.
    
    Args:
        dataframe: pandas DataFrame
    
    Returns:
        Dictionary containing statistics
    """
    numeric_df = dataframe.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return {}
    
    stats = {
        'mean': numeric_df.mean().to_dict(),
        'median': numeric_df.median().to_dict(),
        'std': numeric_df.std().to_dict(),
        'min': numeric_df.min().to_dict(),
        'max': numeric_df.max().to_dict()
    }
    
    return stats

if __name__ == "__main__":
    sample_data = {
        'feature_a': np.random.normal(50, 15, 100),
        'feature_b': np.random.exponential(2, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_statistics(df))
    
    cleaned_df = clean_dataset(df, ['feature_a', 'feature_b'])
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_statistics(cleaned_df))