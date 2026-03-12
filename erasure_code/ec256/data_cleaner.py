
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        clean_data = self.data.copy()
        for col in columns:
            if col in clean_data.columns:
                Q1 = clean_data[col].quantile(0.25)
                Q3 = clean_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                clean_data = clean_data[(clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)]
        return clean_data
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        normalized_data = self.data.copy()
        for col in columns:
            if col in normalized_data.columns:
                min_val = normalized_data[col].min()
                max_val = normalized_data[col].max()
                if max_val != min_val:
                    normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        return normalized_data
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        standardized_data = self.data.copy()
        for col in columns:
            if col in standardized_data.columns:
                mean_val = standardized_data[col].mean()
                std_val = standardized_data[col].std()
                if std_val > 0:
                    standardized_data[col] = (standardized_data[col] - mean_val) / std_val
        return standardized_data
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        filled_data = self.data.copy()
        for col in columns:
            if col in filled_data.columns and filled_data[col].isnull().any():
                if strategy == 'mean':
                    fill_value = filled_data[col].mean()
                elif strategy == 'median':
                    fill_value = filled_data[col].median()
                elif strategy == 'mode':
                    fill_value = filled_data[col].mode()[0]
                else:
                    fill_value = 0
                filled_data[col] = filled_data[col].fillna(fill_value)
        return filled_data
    
    def get_cleaning_report(self):
        report = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object']).columns),
            'missing_values': self.data.isnull().sum().sum(),
            'missing_percentage': (self.data.isnull().sum().sum() / (self.original_shape[0] * self.original_shape[1])) * 100
        }
        return report

def create_sample_data():
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'temperature': np.random.normal(25, 5, 100),
        'humidity': np.random.uniform(30, 90, 100),
        'pressure': np.random.normal(1013, 10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    data.loc[np.random.choice(100, 5), 'temperature'] = np.nan
    data.loc[np.random.choice(100, 3), 'pressure'] = np.nan
    data.loc[10:15, 'temperature'] = data.loc[10:15, 'temperature'] * 3
    return data

if __name__ == "__main__":
    sample_data = create_sample_data()
    cleaner = DataCleaner(sample_data)
    
    print("Data Cleaning Utility")
    print("=" * 50)
    report = cleaner.get_cleaning_report()
    for key, value in report.items():
        print(f"{key}: {value}")
    
    print("\nProcessing data...")
    cleaned_data = cleaner.remove_outliers_iqr(['temperature', 'pressure'])
    filled_data = cleaner.handle_missing_values(strategy='mean')
    normalized_data = cleaner.normalize_minmax(['temperature', 'humidity', 'pressure'])
    
    print(f"\nOriginal data shape: {sample_data.shape}")
    print(f"After outlier removal: {cleaned_data.shape}")
    print(f"Missing values handled: {filled_data.isnull().sum().sum()}")
    print(f"Normalization complete for {len(['temperature', 'humidity', 'pressure'])} columns")