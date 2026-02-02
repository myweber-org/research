
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
            print(f"File not found: {self.file_path}")
            return False
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                missing_count = self.df[col].isnull().sum()
                if missing_count > 0:
                    if strategy == 'mean':
                        fill_value = self.df[col].mean()
                    elif strategy == 'median':
                        fill_value = self.df[col].median()
                    elif strategy == 'mode':
                        fill_value = self.df[col].mode()[0]
                    elif strategy == 'drop':
                        self.df = self.df.dropna(subset=[col])
                        print(f"Dropped rows with missing values in column: {col}")
                        continue
                    else:
                        fill_value = 0
                    
                    self.df[col].fillna(fill_value, inplace=True)
                    print(f"Filled {missing_count} missing values in column '{col}' with {strategy}: {fill_value}")
    
    def remove_duplicates(self):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed = initial_rows - len(self.df)
        print(f"Removed {removed} duplicate rows")
    
    def normalize_column(self, column_name):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        if column_name in self.df.columns:
            col_min = self.df[column_name].min()
            col_max = self.df[column_name].max()
            
            if col_max != col_min:
                self.df[column_name] = (self.df[column_name] - col_min) / (col_max - col_min)
                print(f"Normalized column '{column_name}' to range [0, 1]")
            else:
                print(f"Column '{column_name}' has constant values, skipping normalization")
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to: {output_path}")
        return output_path
    
    def get_summary(self):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        summary = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': len(self.df) - len(self.df.drop_duplicates()),
            'data_types': self.df.dtypes.to_dict()
        }
        
        return summary

def process_csv_file(input_file, output_dir='cleaned_data'):
    cleaner = DataCleaner(input_file)
    
    if cleaner.load_data():
        print("Initial data summary:")
        print(cleaner.get_summary())
        
        cleaner.handle_missing_values(strategy='mean')
        cleaner.remove_duplicates()
        
        numeric_cols = cleaner.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:
            cleaner.normalize_column(col)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        output_path = cleaner.save_cleaned_data(output_dir / f"cleaned_{Path(input_file).name}")
        
        print("\nFinal data summary:")
        print(cleaner.get_summary())
        
        return output_path
    
    return None

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'score': [85, 92, None, 78, 88, 85],
        'category': ['A', 'B', 'A', 'C', 'B', 'A']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = 'test_data.csv'
    test_df.to_csv(test_file, index=False)
    
    result = process_csv_file(test_file)
    
    if result:
        print(f"\nCleaning completed. Output file: {result}")
    
    Path(test_file).unlink(missing_ok=True)
    if result:
        Path(result).unlink(missing_ok=True)