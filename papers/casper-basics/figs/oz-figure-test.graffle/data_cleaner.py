import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mode().iloc[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_value)
        elif strategy == 'drop':
            self.df = self.df.dropna(subset=numeric_cols)
        
        return self
    
    def detect_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_mask = pd.DataFrame(False, index=self.df.index, columns=numeric_cols)
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outliers_mask[col] = z_scores > threshold
        
        self.df['is_outlier'] = outliers_mask.any(axis=1)
        return self
    
    def remove_outliers(self):
        if 'is_outlier' in self.df.columns:
            self.df = self.df[~self.df['is_outlier']].drop(columns=['is_outlier'])
        return self
    
    def normalize_data(self, method='minmax'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        cleaned_shape = self.df.shape
        rows_removed = self.original_shape[0] - cleaned_shape[0]
        cols_removed = self.original_shape[1] - cleaned_shape[1]
        
        summary = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': rows_removed,
            'columns_removed': cols_removed
        }
        
        return summary

def example_usage():
    data = {
        'A': [1, 2, 2, 4, 5, 6, 100],
        'B': [10, 20, 20, 40, 50, 60, 600],
        'C': [100, 200, 200, 400, 500, 600, 6000]
    }
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    cleaned_df = (cleaner
                 .remove_duplicates()
                 .handle_missing_values(strategy='mean')
                 .detect_outliers_zscore(threshold=2.5)
                 .remove_outliers()
                 .normalize_data(method='minmax')
                 .get_cleaned_data())
    
    print("Original data shape:", df.shape)
    print("Cleaned data shape:", cleaned_df.shape)
    print("\nCleaned data:")
    print(cleaned_df)
    
    summary = cleaner.get_summary()
    print("\nCleaning summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    example_usage()