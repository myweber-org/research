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
    
    return cleaner.get_cleaned_data()