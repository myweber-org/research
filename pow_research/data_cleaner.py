import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Loaded data with shape: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return False
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def remove_duplicates(self):
        if self.df is not None:
            initial_rows = len(self.df)
            self.df.drop_duplicates(inplace=True)
            removed = initial_rows - len(self.df)
            print(f"Removed {removed} duplicate rows")
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is not None:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns
            
            for col in columns:
                if col in self.df.columns:
                    if strategy == 'mean':
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif strategy == 'median':
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif strategy == 'mode':
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    elif strategy == 'drop':
                        self.df.dropna(subset=[col], inplace=True)
            
            print(f"Applied {strategy} strategy to handle missing values")
    
    def normalize_numeric(self, columns=None):
        if self.df is not None:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns
            
            for col in columns:
                if col in self.df.columns:
                    col_min = self.df[col].min()
                    col_max = self.df[col].max()
                    if col_max > col_min:
                        self.df[col] = (self.df[col] - col_min) / (col_max - col_min)
            
            print(f"Normalized {len(columns)} numeric columns")
    
    def save_cleaned_data(self, output_path=None):
        if self.df is not None:
            if output_path is None:
                output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
            
            self.df.to_csv(output_path, index=False)
            print(f"Saved cleaned data to {output_path}")
            return output_path
    
    def get_summary(self):
        if self.df is not None:
            summary = {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'missing_values': self.df.isnull().sum().sum(),
                'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(self.df.select_dtypes(include=['object']).columns)
            }
            return summary

def process_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    if cleaner.load_data():
        cleaner.remove_duplicates()
        cleaner.handle_missing_values(strategy='mean')
        cleaner.normalize_numeric()
        
        if output_file:
            result_path = cleaner.save_cleaned_data(output_file)
        else:
            result_path = cleaner.save_cleaned_data()
        
        summary = cleaner.get_summary()
        print(f"Processing complete. Summary: {summary}")
        
        return result_path
    
    return None

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, 20.3, np.nan, 40.7, 50.1, 50.1],
        'category': ['A', 'B', 'A', 'B', 'A', 'A'],
        'score': [85, 92, 78, np.nan, 88, 88]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = 'test_data.csv'
    test_df.to_csv(test_file, index=False)
    
    print("Testing DataCleaner utility...")
    result = process_csv_file(test_file, 'cleaned_test_data.csv')
    
    import os
    if os.path.exists(test_file):
        os.remove(test_file)