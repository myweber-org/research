
import pandas as pd

def clean_dataframe(df):
    """
    Remove rows with null values and standardize column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')
    
    return df_cleaned

def filter_by_threshold(df, column, threshold):
    """
    Filter DataFrame rows where column value exceeds threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to filter by.
        threshold (float): Threshold value.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    return df[df[column] > threshold]

def main():
    # Example usage
    data = {
        'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, None, 35, 40],
        'Score': [85.5, 92.0, 78.5, 88.0]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = clean_dataframe(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print()
    
    filtered_df = filter_by_threshold(cleaned_df, 'score', 80.0)
    print("Filtered DataFrame (score > 80):")
    print(filtered_df)

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    mask = z_scores < threshold
    
    filtered_data = data[mask]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    stats_report = {}
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        original_count = len(cleaned_df)
        
        if outlier_method == 'iqr':
            cleaned_df, removed = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df, removed = remove_outliers_zscore(cleaned_df, col)
        else:
            removed = 0
        
        if normalize_method == 'minmax':
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[f'{col}_normalized'] = normalize_zscore(cleaned_df, col)
        
        stats_report[col] = {
            'original_samples': original_count,
            'removed_outliers': removed,
            'final_samples': len(cleaned_df)
        }
    
    return cleaned_df, stats_report

def validate_data(df, required_columns, min_rows=10):
    """
    Validate dataset structure and content
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'insufficient_rows': False,
        'null_counts': {}
    }
    
    for col in required_columns:
        if col not in df.columns:
            validation_result['missing_columns'].append(col)
            validation_result['is_valid'] = False
    
    if len(df) < min_rows:
        validation_result['insufficient_rows'] = True
        validation_result['is_valid'] = False
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_result['null_counts'][col] = null_count
    
    return validation_result