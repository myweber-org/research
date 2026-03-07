
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
        
        self.df = clean_df
        return self
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
                clean_df = clean_df[(z_scores < threshold) | clean_df[col].isna()]
        
        self.df = clean_df
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'removed_rows': self.get_removed_count(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def load_and_clean_data(filepath, cleaning_steps=None):
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    if cleaning_steps:
        for step in cleaning_steps:
            if hasattr(cleaner, step):
                getattr(cleaner, step)()
    
    return cleaner.get_cleaned_data(), cleaner.get_summary()
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, column, threshold=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        self.df = self.df[mask]
        removed_count = self.original_shape[0] - self.df.shape[0]
        return removed_count
    
    def zscore_normalize(self, column):
        mean_val = self.df[column].mean()
        std_val = self.df[column].std()
        if std_val > 0:
            self.df[f'{column}_normalized'] = (self.df[column] - mean_val) / std_val
            return True
        return False
    
    def minmax_scale(self, column, feature_range=(0, 1)):
        min_val = self.df[column].min()
        max_val = self.df[column].max()
        if max_val > min_val:
            a, b = feature_range
            self.df[f'{column}_scaled'] = a + (self.df[column] - min_val) * (b - a) / (max_val - min_val)
            return True
        return False
    
    def handle_missing(self, column, strategy='mean'):
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = 0
            
        missing_count = self.df[column].isna().sum()
        self.df[column] = self.df[column].fillna(fill_value)
        return missing_count
    
    def get_clean_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': list(self.df.columns),
            'removed_rows': self.original_shape[0] - self.df.shape[0]
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.randint(1, 100, 1000)
    }
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'feature_b'] = 9999
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Initial shape:", cleaner.original_shape)
    
    removed = cleaner.remove_outliers_iqr('feature_b')
    print(f"Removed {removed} outliers from feature_b")
    
    missing_filled = cleaner.handle_missing('feature_a', 'mean')
    print(f"Filled {missing_filled} missing values in feature_a")
    
    cleaner.zscore_normalize('feature_c')
    cleaner.minmax_scale('feature_a')
    
    clean_df = cleaner.get_clean_data()
    summary = cleaner.get_summary()
    
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nFirst 5 rows of cleaned data:")
    print(clean_df.head())