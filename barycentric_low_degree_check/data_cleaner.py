
import pandas as pd
import numpy as np
from pathlib import Path

class CSVDataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Loaded {len(self.df)} rows from {self.file_path.name}")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found")
            return False
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def check_missing_values(self):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return None
        
        missing_counts = self.df.isnull().sum()
        missing_percentage = (missing_counts / len(self.df)) * 100
        
        missing_info = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_percentage
        })
        
        return missing_info[missing_info['missing_count'] > 0]
    
    def fill_missing_numeric(self, strategy='mean'):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
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
                    print(f"Unknown strategy: {strategy}. Using mean.")
                    fill_value = self.df[col].mean()
                
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in {col} with {strategy} value: {fill_value:.2f}")
    
    def fill_missing_categorical(self, strategy='mode'):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if self.df[col].isnull().any():
                if strategy == 'mode':
                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                elif strategy == 'unknown':
                    fill_value = 'Unknown'
                else:
                    print(f"Unknown strategy: {strategy}. Using mode.")
                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in {col} with value: {fill_value}")
    
    def remove_duplicates(self, subset=None, keep='first'):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return 0
        
        initial_count = len(self.df)
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        removed_count = initial_count - len(self.df)
        
        if removed_count > 0:
            print(f"Removed {removed_count} duplicate rows")
        
        return removed_count
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return False
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        try:
            self.df.to_csv(output_path, index=False)
            print(f"Saved cleaned data to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False
    
    def get_summary(self):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return None
        
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        return summary

def clean_csv_file(input_file, output_file=None):
    cleaner = CSVDataCleaner(input_file)
    
    if not cleaner.load_data():
        return False
    
    print("\n--- Initial Data Summary ---")
    summary = cleaner.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n--- Missing Values Analysis ---")
    missing_info = cleaner.check_missing_values()
    if missing_info is not None and len(missing_info) > 0:
        print(missing_info)
        
        cleaner.fill_missing_numeric(strategy='mean')
        cleaner.fill_missing_categorical(strategy='mode')
    else:
        print("No missing values found")
    
    print("\n--- Removing Duplicates ---")
    duplicates_removed = cleaner.remove_duplicates()
    print(f"Total duplicates removed: {duplicates_removed}")
    
    if output_file:
        success = cleaner.save_cleaned_data(output_file)
    else:
        success = cleaner.save_cleaned_data()
    
    if success:
        print("\n--- Final Data Summary ---")
        final_summary = cleaner.get_summary()
        for key, value in final_summary.items():
            print(f"{key}: {value}")
    
    return success

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        clean_csv_file(input_file, output_file)
    else:
        print("Usage: python data_cleaner.py <input_csv> [output_csv]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")