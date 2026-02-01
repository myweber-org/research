
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        
        removed_count = len(self.df) - len(clean_df)
        self.df = clean_df
        return removed_count
    
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in normalized_df.columns:
                if method == 'zscore':
                    normalized_df[col] = stats.zscore(normalized_df[col])
                elif method == 'minmax':
                    min_val = normalized_df[col].min()
                    max_val = normalized_df[col].max()
                    if max_val != min_val:
                        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                elif method == 'robust':
                    median = normalized_df[col].median()
                    iqr = normalized_df[col].quantile(0.75) - normalized_df[col].quantile(0.25)
                    if iqr != 0:
                        normalized_df[col] = (normalized_df[col] - median) / iqr
        
        self.df = normalized_df
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in filled_df.columns and filled_df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = filled_df[col].mean()
                elif strategy == 'median':
                    fill_value = filled_df[col].median()
                elif strategy == 'mode':
                    fill_value = filled_df[col].mode()[0]
                elif strategy == 'drop':
                    filled_df = filled_df.dropna(subset=[col])
                    continue
                else:
                    fill_value = 0
                
                filled_df[col] = filled_df[col].fillna(fill_value)
        
        self.df = filled_df
        return self.df
    
    def get_cleaned_data(self):
        return self.df
    
    def get_cleaning_stats(self):
        return {
            'original_rows': self.original_shape[0],
            'cleaned_rows': len(self.df),
            'rows_removed': self.original_shape[0] - len(self.df),
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1]
        }

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    cleaner.handle_missing_values(strategy='median')
    cleaner.remove_outliers_iqr(factor=1.5)
    cleaner.normalize_data(method='zscore')
    
    stats = cleaner.get_cleaning_stats()
    cleaned_df = cleaner.get_cleaned_data()
    
    return cleaned_df, stats