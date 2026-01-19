
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
            
        clean_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                mask = (clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)
                clean_df = clean_df[mask]
                
        return clean_df.reset_index(drop=True)
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
            
        clean_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                z_scores = np.abs(stats.zscore(clean_df[col].fillna(clean_df[col].mean())))
                mask = z_scores < threshold
                clean_df = clean_df[mask]
                
        return clean_df.reset_index(drop=True)
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        normalized_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                    
        return normalized_df
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        normalized_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val > 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
                    
        return normalized_df
    
    def fill_missing(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        filled_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                if strategy == 'mean':
                    filled_df[col] = filled_df[col].fillna(filled_df[col].mean())
                elif strategy == 'median':
                    filled_df[col] = filled_df[col].fillna(filled_df[col].median())
                elif strategy == 'mode':
                    filled_df[col] = filled_df[col].fillna(filled_df[col].mode()[0])
                elif strategy == 'zero':
                    filled_df[col] = filled_df[col].fillna(0)
                    
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': self.numeric_columns,
            'missing_values': self.df[self.numeric_columns].isnull().sum().to_dict(),
            'basic_stats': self.df[self.numeric_columns].describe().to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    indices = np.random.choice(1000, 50, replace=False)
    for idx in indices:
        col = np.random.choice(['feature_a', 'feature_b', 'feature_c'])
        df.loc[idx, col] = np.nan
    
    outlier_indices = np.random.choice(1000, 20, replace=False)
    for idx in outlier_indices:
        col = np.random.choice(['feature_a', 'feature_b', 'feature_c'])
        df.loc[idx, col] = df.loc[idx, col] * 10
        
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data shape:", cleaner.df.shape)
    print("\nMissing values:")
    print(cleaner.df.isnull().sum())
    
    cleaned_df = cleaner.remove_outliers_iqr()
    print("\nAfter IQR outlier removal:", cleaned_df.shape)
    
    normalized_df = cleaner.normalize_minmax()
    print("\nAfter min-max normalization:")
    print(normalized_df[['feature_a', 'feature_b', 'feature_c']].describe())
    
    filled_df = cleaner.fill_missing(strategy='mean')
    print("\nAfter filling missing values:", filled_df.isnull().sum().sum())