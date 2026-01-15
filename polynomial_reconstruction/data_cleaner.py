
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
    return cleaned_df.reset_index(drop=True)import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
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
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column not in df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, column)
        
        if normalize_method == 'minmax':
            cleaned_df[column] = normalize_minmax(cleaned_df, column)
        elif normalize_method == 'zscore':
            cleaned_df[column] = normalize_zscore(cleaned_df, column)
    
    return cleaned_df

def get_data_summary(df):
    """
    Generate summary statistics for the dataset
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("Data summary:", get_data_summary(sample_data))
    
    cleaned_data = clean_dataset(sample_data, ['feature1', 'feature2'])
    print("Cleaned data shape:", cleaned_data.shape)
    print("Cleaned data summary:", get_data_summary(cleaned_data))