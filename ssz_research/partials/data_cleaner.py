
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        filtered = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        removed_count = len(self.df) - len(filtered)
        self.df = filtered
        return removed_count
    
    def zscore_normalize(self, column):
        mean = self.df[column].mean()
        std = self.df[column].std()
        if std > 0:
            self.df[f'{column}_normalized'] = (self.df[column] - mean) / std
            return True
        return False
    
    def fill_missing_median(self, column):
        median_value = self.df[column].median()
        missing_count = self.df[column].isna().sum()
        self.df[column].fillna(median_value, inplace=True)
        return missing_count
    
    def get_summary(self):
        return {
            'original_rows': self.original_shape[0],
            'current_rows': len(self.df),
            'removed_rows': self.original_shape[0] - len(self.df),
            'columns': list(self.df.columns)
        }
    
    def get_cleaned_data(self):
        return self.df.copy()

def process_dataset(filepath):
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        cleaner.fill_missing_median(col)
        removed = cleaner.remove_outliers_iqr(col)
        if removed > 0:
            print(f"Removed {removed} outliers from {col}")
        cleaner.zscore_normalize(col)
    
    summary = cleaner.get_summary()
    print(f"Data cleaning complete. Removed {summary['removed_rows']} rows total.")
    
    return cleaner.get_cleaned_data()