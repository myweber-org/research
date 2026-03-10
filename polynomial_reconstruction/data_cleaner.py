
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.data.columns
            
        clean_data = self.data.copy()
        for col in columns:
            if clean_data[col].dtype in ['int64', 'float64']:
                Q1 = clean_data[col].quantile(0.25)
                Q3 = clean_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_data = clean_data[(clean_data[col] >= lower_bound) & 
                                       (clean_data[col] <= upper_bound)]
        return clean_data
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        normalized_data = self.data.copy()
        for col in columns:
            if normalized_data[col].dtype in ['int64', 'float64']:
                min_val = normalized_data[col].min()
                max_val = normalized_data[col].max()
                if max_val > min_val:
                    normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        return normalized_data
    
    def standardize_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        standardized_data = self.data.copy()
        for col in columns:
            if standardized_data[col].dtype in ['int64', 'float64']:
                mean_val = standardized_data[col].mean()
                std_val = standardized_data[col].std()
                if std_val > 0:
                    z_scores = np.abs((standardized_data[col] - mean_val) / std_val)
                    standardized_data = standardized_data[z_scores < threshold]
        return standardized_data
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.data.columns
            
        filled_data = self.data.copy()
        for col in columns:
            if filled_data[col].isnull().any():
                if strategy == 'mean' and filled_data[col].dtype in ['int64', 'float64']:
                    fill_value = filled_data[col].mean()
                elif strategy == 'median' and filled_data[col].dtype in ['int64', 'float64']:
                    fill_value = filled_data[col].median()
                elif strategy == 'mode':
                    fill_value = filled_data[col].mode()[0]
                elif strategy == 'drop':
                    filled_data = filled_data.dropna(subset=[col])
                    continue
                else:
                    fill_value = 0
                filled_data[col] = filled_data[col].fillna(fill_value)
        return filled_data
    
    def get_cleaning_report(self):
        report = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'missing_values': self.data.isnull().sum().sum(),
            'numeric_columns': len(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.data.select_dtypes(include=['object']).columns)
        }
        return report

def create_sample_dataset():
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
    return data

if __name__ == "__main__":
    sample_data = create_sample_dataset()
    cleaner = DataCleaner(sample_data)
    
    print("Data Cleaning Utility")
    print("=" * 50)
    report = cleaner.get_cleaning_report()
    for key, value in report.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    cleaned_data = cleaner.handle_missing_values(strategy='mean')
    normalized_data = cleaner.normalize_minmax(['temperature', 'humidity', 'pressure'])
    
    print(f"\nOriginal data shape: {sample_data.shape}")
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Normalized data sample:\n{normalized_data[['temperature', 'humidity', 'pressure']].head()}")