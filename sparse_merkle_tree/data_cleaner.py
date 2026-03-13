import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
    def detect_outliers_zscore(self, threshold=3):
        outliers = {}
        for col in self.numeric_columns:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outlier_indices = np.where(z_scores > threshold)[0]
            if len(outlier_indices) > 0:
                outliers[col] = self.df.iloc[outlier_indices].index.tolist()
        return outliers
    
    def impute_missing_median(self):
        for col in self.numeric_columns:
            if self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
        return self.df
    
    def remove_duplicates(self, subset=None, keep='first'):
        initial_count = len(self.df)
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        removed = initial_count - len(self.df)
        return self.df, removed
    
    def normalize_data(self, method='minmax'):
        normalized_df = self.df.copy()
        for col in self.numeric_columns:
            if method == 'minmax':
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val > 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        return normalized_df
    
    def get_cleaned_data(self):
        return self.df
import pandas as pd
import numpy as np

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

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
        df = normalize_minmax(df, col)
    
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv')
    cleaned_data.to_csv('cleaned_data.csv', index=False)import numpy as np
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
    if max_val - min_val == 0:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.dropna()

def calculate_statistics(df):
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'median': df[col].median(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    return stats

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['X', 'Y', 'Z'], 200)
    })
    
    numeric_cols = ['feature_a', 'feature_b']
    cleaned = clean_dataset(sample_data, numeric_cols)
    statistics = calculate_statistics(cleaned)
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print("Statistics:", statistics)
import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column]
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_data(data, column):
    """
    Standardize data using Z-score normalization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values with specified strategy
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for col in columns:
        if data_copy[col].isnull().any():
            if strategy == 'mean':
                fill_value = data_copy[col].mean()
            elif strategy == 'median':
                fill_value = data_copy[col].median()
            elif strategy == 'mode':
                fill_value = data_copy[col].mode()[0]
            elif strategy == 'ffill':
                data_copy[col] = data_copy[col].fillna(method='ffill')
                continue
            elif strategy == 'bfill':
                data_copy[col] = data_copy[col].fillna(method='bfill')
                continue
            else:
                fill_value = 0
            
            data_copy[col] = data_copy[col].fillna(fill_value)
    
    return data_copy

def clean_dataset(data, outlier_method='zscore', normalize=False, standardize=False, missing_strategy='mean'):
    """
    Main function to clean dataset with multiple options
    """
    cleaned_data = data.copy()
    
    numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, col)
        elif outlier_method == 'iqr':
            outliers = detect_outliers_iqr(cleaned_data, col)
            cleaned_data = cleaned_data.drop(outliers.index)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    if normalize:
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
    
    if standardize:
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = standardize_data(cleaned_data, col)
    
    return cleaned_data.reset_index(drop=True)