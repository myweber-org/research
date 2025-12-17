import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def impute_missing_values(data, column, method='mean'):
    if method == 'mean':
        fill_value = data[column].mean()
    elif method == 'median':
        fill_value = data[column].median()
    elif method == 'mode':
        fill_value = data[column].mode()[0]
    else:
        fill_value = 0
    
    data[column] = data[column].fillna(fill_value)
    return data

def remove_duplicates(data, subset=None):
    if subset:
        return data.drop_duplicates(subset=subset)
    return data.drop_duplicates()

def standardize_column(data, column):
    mean = data[column].mean()
    std = data[column].std()
    data[column] = (data[column] - mean) / std
    return data

def clean_dataset(data, config):
    cleaned_data = data.copy()
    
    for column in config.get('outlier_columns', []):
        outliers = detect_outliers_iqr(cleaned_data, column)
        if not outliers.empty:
            cleaned_data = cleaned_data.drop(outliers.index)
    
    for column, method in config.get('impute_columns', {}).items():
        cleaned_data = impute_missing_values(cleaned_data, column, method)
    
    if config.get('remove_duplicates', False):
        cleaned_data = remove_duplicates(cleaned_data, config.get('duplicate_subset'))
    
    for column in config.get('standardize_columns', []):
        cleaned_data = standardize_column(cleaned_data, column)
    
    return cleaned_data

def main():
    sample_data = pd.DataFrame({
        'A': [1, 2, 3, 100, 5, np.nan, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': ['x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y']
    })
    
    config = {
        'outlier_columns': ['A'],
        'impute_columns': {'A': 'mean'},
        'remove_duplicates': True,
        'duplicate_subset': ['C'],
        'standardize_columns': ['B']
    }
    
    cleaned = clean_dataset(sample_data, config)
    print("Original data shape:", sample_data.shape)
    print("Cleaned data shape:", cleaned.shape)
    print("\nCleaned data:")
    print(cleaned)

if __name__ == "__main__":
    main()