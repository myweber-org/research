
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
        
        return clean_df
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
                mask = z_scores < threshold
                clean_df = clean_df[mask]
        
        return clean_df
    
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
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        filled_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                mean_val = filled_df[col].mean()
                filled_df[col] = filled_df[col].fillna(mean_val)
        
        return filled_df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        filled_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                median_val = filled_df[col].median()
                filled_df[col] = filled_df[col].fillna(median_val)
        
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': self.numeric_columns,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    data['feature_a'][[10, 25, 50]] = np.nan
    data['feature_b'][[15, 30, 75]] = 1000
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_sample_data()
    cleaner = DataCleaner(df)
    
    print("Data Summary:")
    print(f"Original shape: {cleaner.get_summary()['original_shape']}")
    print(f"Numeric columns: {cleaner.get_summary()['numeric_columns']}")
    print(f"Missing values: {cleaner.get_summary()['missing_values']}")
    
    cleaned_df = cleaner.remove_outliers_iqr(['feature_b'])
    normalized_df = cleaner.normalize_minmax()
    filled_df = cleaner.fill_missing_mean()
    
    print(f"\nCleaned shape: {cleaned_df.shape}")
    print(f"Normalized shape: {normalized_df.shape}")
    print(f"Filled shape: {filled_df.shape}")