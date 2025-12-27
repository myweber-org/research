
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def process_numerical_data(df, columns):
    """
    Process multiple numerical columns by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Processed DataFrame
    """
    processed_df = df.copy()
    
    for col in columns:
        if col in processed_df.columns and pd.api.types.is_numeric_dtype(processed_df[col]):
            processed_df = remove_outliers_iqr(processed_df, col)
    
    return processed_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'pressure': [1013, 1014, 1015, 1016, 1017, 2000, 1018, 1019, 1020, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nSummary statistics before cleaning:")
    for col in ['temperature', 'pressure']:
        stats = calculate_summary_statistics(df, col)
        print(f"{col}: {stats}")
    
    cleaned_df = process_numerical_data(df, ['temperature', 'pressure'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nSummary statistics after cleaning:")
    for col in ['temperature', 'pressure']:
        stats = calculate_summary_statistics(cleaned_df, col)
        print(f"{col}: {stats}")
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def clean_numeric_data(values, default=0):
    """
    Clean numeric data by converting strings to floats.
    Non-numeric values are replaced with the default value.
    """
    cleaned = []
    for val in values:
        try:
            cleaned.append(float(val))
        except (ValueError, TypeError):
            cleaned.append(default)
    return cleaned

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))
    
    numeric_mixed = [1, "2.5", "invalid", 3.7, None]
    print("Mixed data:", numeric_mixed)
    print("Cleaned numeric:", clean_numeric_data(numeric_mixed))import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process (None for all numeric columns)
        factor: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None):
    """
    Normalize data using min-max scaling.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize (None for all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def remove_missing_rows(df, threshold=0.8):
    """
    Remove rows with excessive missing values.
    
    Args:
        df: pandas DataFrame
        threshold: maximum allowed missing value ratio per row (0-1)
    
    Returns:
        DataFrame with rows removed
    """
    missing_ratio = df.isnull().sum(axis=1) / df.shape[1]
    mask = missing_ratio <= threshold
    return df[mask].reset_index(drop=True)

def clean_dataset(df, outlier_columns=None, normalize_columns=None, missing_threshold=0.8):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        outlier_columns: columns for outlier removal
        normalize_columns: columns for normalization
        missing_threshold: threshold for missing value removal
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    df_clean = remove_missing_rows(df_clean, threshold=missing_threshold)
    df_clean = remove_outliers_iqr(df_clean, columns=outlier_columns)
    df_clean = normalize_minmax(df_clean, columns=normalize_columns)
    
    return df_clean

def validate_data(df, check_duplicates=True, check_types=True):
    """
    Validate data quality.
    
    Args:
        df: pandas DataFrame
        check_duplicates: flag to check for duplicate rows
        check_types: flag to validate data types
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'unique_rows': len(df.drop_duplicates())
    }
    
    if check_duplicates:
        validation_results['duplicate_rows'] = len(df) - len(df.drop_duplicates())
    
    if check_types:
        type_counts = df.dtypes.value_counts().to_dict()
        validation_results['data_types'] = type_counts
    
    return validation_results

def generate_summary(df):
    """
    Generate statistical summary of the dataset.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    if len(df.select_dtypes(include=[np.number]).columns) > 0:
        numeric_stats = df.select_dtypes(include=[np.number]).describe().to_dict()
        summary['numeric_statistics'] = numeric_stats
    
    return summary