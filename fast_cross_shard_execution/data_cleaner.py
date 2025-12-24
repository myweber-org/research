
def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset)
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self.df
    
    def handle_missing_values(self, strategy: str = 'drop', fill_value: Optional[float] = None) -> pd.DataFrame:
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                self.df = self.df.fillna(self.df.mean())
        else:
            raise ValueError("Strategy must be 'drop' or 'fill'")
        
        print(f"Missing values handled using '{strategy}' strategy")
        return self.df
    
    def normalize_column(self, column: str) -> pd.DataFrame:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        col_min = self.df[column].min()
        col_max = self.df[column].max()
        
        if col_max != col_min:
            self.df[column] = (self.df[column] - col_min) / (col_max - col_min)
            print(f"Column '{column}' normalized to range [0, 1]")
        else:
            print(f"Column '{column}' has constant values, skipping normalization")
        
        return self.df
    
    def get_cleaning_report(self) -> dict:
        final_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'final_rows': final_shape[0],
            'final_columns': final_shape[1],
            'rows_removed': self.original_shape[0] - final_shape[0],
            'null_count': self.df.isnull().sum().sum()
        }

def create_sample_data() -> pd.DataFrame:
    data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10.5, np.nan, 15.2, 20.1, 25.0, 25.0, np.nan],
        'category': ['A', 'B', 'A', 'C', 'B', 'B', 'A']
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_sample_data()
    cleaner = DataCleaner(df)
    
    print("Initial data shape:", df.shape)
    cleaner.remove_duplicates(subset=['id'])
    cleaner.handle_missing_values(strategy='fill', fill_value=0)
    cleaner.normalize_column('value')
    
    report = cleaner.get_cleaning_report()
    print("\nCleaning Report:")
    for key, value in report.items():
        print(f"{key}: {value}")