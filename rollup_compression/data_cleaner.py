
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        self.df = pd.read_csv(self.file_path)
        print(f"Loaded data with shape: {self.df.shape}")
        return self.df
    
    def check_missing_values(self):
        if self.df is None:
            self.load_data()
        
        missing_counts = self.df.isnull().sum()
        missing_percentage = (missing_counts / len(self.df)) * 100
        
        missing_info = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_percentage
        })
        
        return missing_info[missing_info['missing_count'] > 0]
    
    def fill_missing_numeric(self, strategy='mean'):
        if self.df is None:
            self.load_data()
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError("Strategy must be 'mean', 'median', or 'zero'")
                
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in {col} with {strategy}: {fill_value}")
        
        return self.df
    
    def fill_missing_categorical(self, strategy='mode'):
        if self.df is None:
            self.load_data()
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if self.df[col].isnull().any():
                if strategy == 'mode':
                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                elif strategy == 'unknown':
                    fill_value = 'Unknown'
                else:
                    raise ValueError("Strategy must be 'mode' or 'unknown'")
                
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in {col} with {strategy}: {fill_value}")
        
        return self.df
    
    def remove_duplicates(self, subset=None, keep='first'):
        if self.df is None:
            self.load_data()
        
        initial_count = len(self.df)
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        removed_count = initial_count - len(self.df)
        
        print(f"Removed {removed_count} duplicate rows")
        return self.df
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            raise ValueError("No data to save. Please load and clean data first.")
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to: {output_path}")
        return output_path

def process_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    try:
        cleaner.load_data()
        
        missing_report = cleaner.check_missing_values()
        if not missing_report.empty:
            print("Missing values found:")
            print(missing_report)
            
            cleaner.fill_missing_numeric(strategy='mean')
            cleaner.fill_missing_categorical(strategy='mode')
        
        cleaner.remove_duplicates()
        
        output_path = cleaner.save_cleaned_data(output_file)
        return output_path
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    Returns a filtered DataFrame.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate basic summary statistics for a column.
    """
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column])
    }
    return stats