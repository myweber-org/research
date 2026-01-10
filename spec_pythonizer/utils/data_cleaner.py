
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def detect_outliers_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        self.df = self.df[(z_scores < threshold) | self.df[column].isna()]
        return self
    
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            self.df[column] = (self.df[column] - mean_val) / std_val
        return self
    
    def fill_missing(self, column, strategy='mean'):
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = strategy
            
        self.df[column].fillna(fill_value, inplace=True)
        return self
    
    def get_cleaned_data(self):
        print(f"Original shape: {self.original_shape}")
        print(f"Cleaned shape: {self.df.shape}")
        print(f"Rows removed: {self.original_shape[0] - self.df.shape[0]}")
        return self.df
    
    def summary(self):
        summary_stats = self.df.describe(include='all')
        missing_values = self.df.isnull().sum()
        print("Summary Statistics:")
        print(summary_stats)
        print("\nMissing Values:")
        print(missing_values)
        return summary_stats, missing_values

def example_usage():
    np.random.seed(42)
    data = {
        'age': np.random.normal(30, 10, 100),
        'income': np.random.exponential(50000, 100),
        'score': np.random.uniform(0, 100, 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'age'] = np.nan
    df.loc[5, 'income'] = 1000000
    
    cleaner = DataCleaner(df)
    cleaner.summary()
    
    outliers = cleaner.detect_outliers_iqr('income')
    print(f"Outliers in income: {len(outliers)}")
    
    cleaner.remove_outliers_zscore('income', threshold=3)
    cleaner.fill_missing('age', strategy='median')
    cleaner.normalize_column('score', method='minmax')
    
    cleaned_df = cleaner.get_cleaned_data()
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(result.head())