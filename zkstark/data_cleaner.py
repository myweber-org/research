import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_missing(self, threshold=0.8):
        self.df = self.df.dropna(thresh=threshold * len(self.df.columns))
        return self
    
    def fill_numeric_missing(self, method='median'):
        for col in self.numeric_columns:
            if self.df[col].isnull().any():
                if method == 'median':
                    fill_value = self.df[col].median()
                elif method == 'mean':
                    fill_value = self.df[col].mean()
                else:
                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                self.df[col] = self.df[col].fillna(fill_value)
        return self
    
    def remove_outliers(self, z_threshold=3):
        for col in self.numeric_columns:
            z_scores = np.abs(stats.zscore(self.df[col]))
            self.df = self.df[z_scores < z_threshold]
        return self
    
    def normalize_numeric(self, method='minmax'):
        for col in self.numeric_columns:
            if method == 'minmax':
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        return self
    
    def get_cleaned_data(self):
        return self.df

def clean_dataset(df, missing_threshold=0.8, outlier_threshold=3):
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                  .remove_missing(missing_threshold)
                  .fill_numeric_missing('median')
                  .remove_outliers(outlier_threshold)
                  .normalize_numeric('minmax')
                  .get_cleaned_data())
    return cleaned_df