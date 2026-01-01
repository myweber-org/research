import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_missing(self, threshold=0.8):
        """Remove columns with missing values above threshold"""
        missing_ratio = self.df.isnull().sum() / len(self.df)
        columns_to_drop = missing_ratio[missing_ratio > threshold].index
        self.df = self.df.drop(columns=columns_to_drop)
        return self
    
    def fill_numeric_missing(self, method='median'):
        """Fill missing values in numeric columns"""
        for col in self.numeric_columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if method == 'median':
                    fill_value = self.df[col].median()
                elif method == 'mean':
                    fill_value = self.df[col].mean()
                elif method == 'mode':
                    fill_value = self.df[col].mode()[0]
                else:
                    fill_value = 0
                self.df[col] = self.df[col].fillna(fill_value)
        return self
    
    def remove_outliers_zscore(self, threshold=3):
        """Remove outliers using z-score method"""
        z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
        outlier_mask = (z_scores < threshold).all(axis=1)
        self.df = self.df[outlier_mask]
        return self
    
    def normalize_numeric(self, method='minmax'):
        """Normalize numeric columns"""
        for col in self.numeric_columns:
            if col in self.df.columns:
                if method == 'minmax':
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val > min_val:
                        self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                elif method == 'standard':
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    if std_val > 0:
                        self.df[col] = (self.df[col] - mean_val) / std_val
        return self
    
    def get_cleaned_data(self):
        """Return cleaned dataframe"""
        return self.df
    
    def save_cleaned_data(self, filepath):
        """Save cleaned data to file"""
        self.df.to_csv(filepath, index=False)

def clean_dataset(df, config=None):
    """Convenience function for quick cleaning"""
    if config is None:
        config = {
            'missing_threshold': 0.8,
            'fill_method': 'median',
            'outlier_threshold': 3,
            'normalize_method': 'minmax'
        }
    
    cleaner = DataCleaner(df)
    cleaner.remove_missing(config['missing_threshold'])
    cleaner.fill_numeric_missing(config['fill_method'])
    cleaner.remove_outliers_zscore(config['outlier_threshold'])
    cleaner.normalize_numeric(config['normalize_method'])
    
    return cleaner.get_cleaned_data()import numpy as np
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
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
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

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    removal_stats = {}
    
    for col in numeric_columns:
        if col not in cleaned_data.columns:
            continue
            
        # Remove outliers
        if outlier_method == 'iqr':
            cleaned_data, removed = remove_outliers_iqr(cleaned_data, col)
        elif outlier_method == 'zscore':
            cleaned_data, removed = remove_outliers_zscore(cleaned_data, col)
        else:
            raise ValueError("Invalid outlier method. Use 'iqr' or 'zscore'")
        
        removal_stats[col] = removed
        
        # Normalize data
        if normalize_method == 'minmax':
            cleaned_data[col] = normalize_minmax(cleaned_data, col)
        elif normalize_method == 'zscore':
            cleaned_data[col] = normalize_zscore(cleaned_data, col)
        else:
            raise ValueError("Invalid normalize method. Use 'minmax' or 'zscore'")
    
    return cleaned_data, removal_stats

def validate_data(data, required_columns, numeric_threshold=0.8):
    """
    Validate dataset structure and content
    """
    validation_results = {
        'missing_columns': [],
        'low_numeric_ratio': [],
        'high_missing_values': []
    }
    
    # Check for required columns
    for col in required_columns:
        if col not in data.columns:
            validation_results['missing_columns'].append(col)
    
    # Check numeric content ratio
    for col in data.select_dtypes(include=[np.number]).columns:
        numeric_ratio = data[col].notna().sum() / len(data)
        if numeric_ratio < numeric_threshold:
            validation_results['low_numeric_ratio'].append((col, numeric_ratio))
    
    # Check for high missing values
    for col in data.columns:
        missing_ratio = data[col].isna().sum() / len(data)
        if missing_ratio > 0.3:
            validation_results['high_missing_values'].append((col, missing_ratio))
    
    return validation_results