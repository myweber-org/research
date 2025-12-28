import numpy as np
import pandas as pd

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def remove_outliers(data, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data to range [0, 1] using min-max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using z-score standardization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns, outlier_threshold=1.5, normalization_method='minmax'):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    """
    cleaned_data = data.copy()
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data = remove_outliers(cleaned_data, col, outlier_threshold)
            if normalization_method == 'minmax':
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
            elif normalization_method == 'zscore':
                cleaned_data[col] = normalize_zscore(cleaned_data, col)
    return cleaned_data

def summary_statistics(data, numeric_columns):
    """
    Generate summary statistics for numeric columns.
    """
    stats = {}
    for col in numeric_columns:
        if col in data.columns:
            stats[col] = {
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'count': data[col].count(),
                'missing': data[col].isnull().sum()
            }
    return pd.DataFrame(stats).T

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    numeric_cols = ['feature1', 'feature2']
    
    print("Original data shape:", sample_data.shape)
    print("\nOriginal summary statistics:")
    print(summary_statistics(sample_data, numeric_cols))
    
    cleaned_data = clean_dataset(sample_data, numeric_cols, outlier_threshold=1.5, normalization_method='minmax')
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("\nCleaned summary statistics:")
    print(summary_statistics(cleaned_data, numeric_cols))