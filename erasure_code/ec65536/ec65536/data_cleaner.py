
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.cleaned_data = None
        
    def remove_outliers_iqr(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.data[column]))
        return self.data[z_scores < threshold]
    
    def normalize_minmax(self, column):
        min_val = self.data[column].min()
        max_val = self.data[column].max()
        return (self.data[column] - min_val) / (max_val - min_val)
    
    def standardize_zscore(self, column):
        mean_val = self.data[column].mean()
        std_val = self.data[column].std()
        return (self.data[column] - mean_val) / std_val
    
    def handle_missing_mean(self, column):
        mean_val = self.data[column].mean()
        return self.data[column].fillna(mean_val)
    
    def handle_missing_median(self, column):
        median_val = self.data[column].median()
        return self.data[column].fillna(median_val)
    
    def clean_pipeline(self, operations):
        temp_data = self.data.copy()
        for operation in operations:
            if operation['type'] == 'remove_outliers':
                method = operation.get('method', 'iqr')
                column = operation['column']
                if method == 'iqr':
                    temp_data = self.remove_outliers_iqr(column)
                elif method == 'zscore':
                    temp_data = self.remove_outliers_zscore(column, operation.get('threshold', 3))
            elif operation['type'] == 'normalize':
                column = operation['column']
                temp_data[column] = self.normalize_minmax(column)
            elif operation['type'] == 'standardize':
                column = operation['column']
                temp_data[column] = self.standardize_zscore(column)
            elif operation['type'] == 'handle_missing':
                column = operation['column']
                method = operation.get('method', 'mean')
                if method == 'mean':
                    temp_data[column] = self.handle_missing_mean(column)
                elif method == 'median':
                    temp_data[column] = self.handle_missing_median(column)
        self.cleaned_data = temp_data
        return self.cleaned_data
    
    def get_summary(self):
        if self.cleaned_data is None:
            return "No cleaning operations performed yet."
        
        summary = {
            'original_rows': len(self.data),
            'cleaned_rows': len(self.cleaned_data),
            'removed_rows': len(self.data) - len(self.cleaned_data),
            'original_columns': list(self.data.columns),
            'cleaned_columns': list(self.cleaned_data.columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000)
    })
    
    cleaner = DataCleaner(data)
    
    operations = [
        {'type': 'remove_outliers', 'method': 'iqr', 'column': 'feature1'},
        {'type': 'remove_outliers', 'method': 'zscore', 'column': 'feature2', 'threshold': 3},
        {'type': 'normalize', 'column': 'feature3'},
        {'type': 'standardize', 'column': 'feature1'}
    ]
    
    cleaned_data = cleaner.clean_pipeline(operations)
    summary = cleaner.get_summary()
    
    print(f"Original data shape: {data.shape}")
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Summary: {summary}")
    
    return cleaned_data

if __name__ == "__main__":
    example_usage()
def deduplicate_list(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result