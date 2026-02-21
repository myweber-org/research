import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            for col in self.numerical_cols:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in self.numerical_cols:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in self.numerical_cols:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        elif strategy == 'constant':
            if fill_value is not None:
                self.df.fillna(fill_value, inplace=True)
        
        for col in self.categorical_cols:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        return self.df
    
    def remove_outliers(self, method='zscore', threshold=3):
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(self.df[self.numerical_cols]))
            mask = (z_scores < threshold).all(axis=1)
            self.df = self.df[mask]
        elif method == 'iqr':
            for col in self.numerical_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        return self.df
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def summary(self):
        missing = self.df.isnull().sum()
        outliers = {}
        
        for col in self.numerical_cols:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outliers[col] = (z_scores > 3).sum()
        
        return {
            'missing_values': missing.to_dict(),
            'outliers_detected': outliers,
            'shape': self.df.shape
        }

def example_usage():
    data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['a', 'b', 'a', 'b', np.nan, 'a']
    }
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    print("Original data:")
    print(df)
    print("\nMissing values summary:")
    print(df.isnull().sum())
    
    cleaned_df = cleaner.handle_missing_values(strategy='mean')
    cleaned_df = cleaner.remove_outliers(method='zscore', threshold=3)
    
    print("\nCleaned data:")
    print(cleaned_df)
    print("\nCleaning summary:")
    print(cleaner.summary())
    
    return cleaned_df

if __name__ == "__main__":
    example_usage()