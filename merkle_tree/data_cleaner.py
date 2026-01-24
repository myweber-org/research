
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling to range [0, 1]
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy
    """
    if columns is None:
        columns = data.columns
    
    data_clean = data.copy()
    
    for column in columns:
        if column not in data.columns:
            continue
            
        if data[column].isnull().any():
            if strategy == 'mean':
                fill_value = data[column].mean()
            elif strategy == 'median':
                fill_value = data[column].median()
            elif strategy == 'mode':
                fill_value = data[column].mode()[0]
            elif strategy == 'drop':
                data_clean = data_clean.dropna(subset=[column])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_clean[column] = data[column].fillna(fill_value)
    
    return data_clean

def create_sample_data():
    """
    Create sample data for testing
    """
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'feature_a': np.random.normal(50, 15, n_samples),
        'feature_b': np.random.exponential(2, n_samples),
        'feature_c': np.random.uniform(0, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=10, replace=False)
    data.loc[missing_indices, 'feature_a'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(n_samples, size=5, replace=False)
    data.loc[outlier_indices, 'feature_b'] *= 10
    
    return data

def main():
    """
    Demonstrate the data cleaning functions
    """
    print("Creating sample data...")
    data = create_sample_data()
    print(f"Original data shape: {data.shape}")
    print(f"Missing values:\n{data.isnull().sum()}")
    
    print("\nHandling missing values...")
    data_clean = handle_missing_values(data, strategy='mean')
    print(f"After cleaning shape: {data_clean.shape}")
    
    print("\nRemoving outliers from feature_b...")
    data_no_outliers, removed = remove_outliers_iqr(data_clean, 'feature_b')
    print(f"Outliers removed: {removed}")
    print(f"Data shape after outlier removal: {data_no_outliers.shape}")
    
    print("\nNormalizing feature_a...")
    data_no_outliers['feature_a_normalized'] = normalize_minmax(data_no_outliers, 'feature_a')
    print(f"Normalized range: [{data_no_outliers['feature_a_normalized'].min():.3f}, "
          f"{data_no_outliers['feature_a_normalized'].max():.3f}]")
    
    print("\nStandardizing feature_c...")
    data_no_outliers['feature_c_standardized'] = standardize_zscore(data_no_outliers, 'feature_c')
    print(f"Standardized mean: {data_no_outliers['feature_c_standardized'].mean():.3f}, "
          f"std: {data_no_outliers['feature_c_standardized'].std():.3f}")
    
    return data_no_outliers

if __name__ == "__main__":
    cleaned_data = main()
    print(f"\nFinal cleaned data shape: {cleaned_data.shape}")
    print("Data cleaning completed successfully.")