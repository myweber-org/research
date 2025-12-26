
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        self.df = self.df[mask]
        return self
        
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        mask = z_scores < threshold
        self.df = self.df[mask]
        return self
        
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
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
            
        self.df[column] = self.df[column].fillna(fill_value)
        return self
        
    def drop_duplicates(self, subset=None, keep='first'):
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self
        
    def get_cleaned_data(self):
        return self.df.copy()
        
    def get_summary(self):
        summary = {
            'original_rows': len(self.df),
            'cleaned_rows': len(self.df),
            'columns': self.original_columns,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = {
        'date': dates,
        'temperature': np.random.normal(25, 5, 100),
        'humidity': np.random.normal(60, 15, 100),
        'pressure': np.random.normal(1013, 10, 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'temperature'] = np.nan
    df.loc[20:25, 'humidity'] = np.nan
    df.loc[5, 'temperature'] = 100
    df.loc[6, 'humidity'] = 200
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    cleaned_df = (cleaner
                 .remove_outliers_iqr('temperature')
                 .remove_outliers_zscore('humidity')
                 .fill_missing('temperature', 'mean')
                 .fill_missing('humidity', 'median')
                 .normalize_column('pressure', 'minmax')
                 .drop_duplicates()
                 .get_cleaned_data())
    
    print("Data cleaning completed successfully")
    print(f"Original shape: {sample_df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print("\nSummary statistics:")
    print(cleaned_df.describe())
import re

def clean_string(text):
    """
    Clean a string by removing extra whitespace and converting to lowercase.
    
    Args:
        text (str): The input string to clean.
    
    Returns:
        str: The cleaned string.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    # Remove leading and trailing whitespace
    cleaned = text.strip()
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Convert to lowercase
    cleaned = cleaned.lower()
    
    return cleaned