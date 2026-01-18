
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.cleaned_data = None
        
    def remove_outliers_iqr(self, columns=None):
        if columns is None:
            columns = self.data.columns
            
        cleaned = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                Q1 = cleaned[col].quantile(0.25)
                Q3 = cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                cleaned = cleaned[(cleaned[col] >= lower_bound) & (cleaned[col] <= upper_bound)]
        self.cleaned_data = cleaned
        return self
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.data.columns
            
        cleaned = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                z_scores = np.abs(stats.zscore(cleaned[col]))
                cleaned = cleaned[z_scores < threshold]
        self.cleaned_data = cleaned
        return self
    
    def normalize_minmax(self, columns=None):
        if self.cleaned_data is None:
            cleaned = self.data.copy()
        else:
            cleaned = self.cleaned_data.copy()
            
        if columns is None:
            columns = cleaned.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                min_val = cleaned[col].min()
                max_val = cleaned[col].max()
                if max_val > min_val:
                    cleaned[col] = (cleaned[col] - min_val) / (max_val - min_val)
        self.cleaned_data = cleaned
        return self
    
    def normalize_zscore(self, columns=None):
        if self.cleaned_data is None:
            cleaned = self.data.copy()
        else:
            cleaned = self.cleaned_data.copy()
            
        if columns is None:
            columns = cleaned.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                mean_val = cleaned[col].mean()
                std_val = cleaned[col].std()
                if std_val > 0:
                    cleaned[col] = (cleaned[col] - mean_val) / std_val
        self.cleaned_data = cleaned
        return self
    
    def fill_missing_mean(self, columns=None):
        if self.cleaned_data is None:
            cleaned = self.data.copy()
        else:
            cleaned = self.cleaned_data.copy()
            
        if columns is None:
            columns = cleaned.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
        self.cleaned_data = cleaned
        return self
    
    def fill_missing_median(self, columns=None):
        if self.cleaned_data is None:
            cleaned = self.data.copy()
        else:
            cleaned = self.cleaned_data.copy()
            
        if columns is None:
            columns = cleaned.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        self.cleaned_data = cleaned
        return self
    
    def get_cleaned_data(self):
        if self.cleaned_data is None:
            return self.data.copy()
        return self.cleaned_data.copy()
    
    def get_summary(self):
        if self.cleaned_data is None:
            df = self.data
        else:
            df = self.cleaned_data
            
        summary = {
            'original_rows': len(self.data),
            'cleaned_rows': len(df),
            'removed_rows': len(self.data) - len(df),
            'missing_values': df.isnull().sum().sum(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(exclude=[np.number]).columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.exponential(1, 100),
        'feature3': np.random.randint(0, 10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    data.loc[10:15, 'feature1'] = np.nan
    data.loc[20:25, 'feature2'] = 100
    
    cleaner = DataCleaner(data)
    result = (cleaner
              .remove_outliers_iqr(['feature1', 'feature2'])
              .fill_missing_mean(['feature1', 'feature2'])
              .normalize_minmax(['feature1', 'feature2'])
              .get_cleaned_data())
    
    summary = cleaner.get_summary()
    return result, summary

if __name__ == "__main__":
    cleaned_data, summary = example_usage()
    print("Data cleaning completed successfully")
    print(f"Original rows: {summary['original_rows']}")
    print(f"Cleaned rows: {summary['cleaned_rows']}")
    print(f"Removed rows: {summary['removed_rows']}")
    print(f"Missing values in cleaned data: {summary['missing_values']}")