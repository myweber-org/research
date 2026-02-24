
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self
        
    def standardize_columns(self) -> 'DataCleaner':
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
        return self
        
    def fill_missing_numeric(self, strategy: str = 'mean') -> 'DataCleaner':
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            fill_values = self.df[numeric_cols].mean()
        elif strategy == 'median':
            fill_values = self.df[numeric_cols].median()
        elif strategy == 'zero':
            fill_values = 0
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
            
        self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_values)
        return self
        
    def remove_outliers_iqr(self, column: str, multiplier: float = 1.5) -> 'DataCleaner':
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found in dataframe")
            
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df.copy()
        
    def get_cleaning_report(self) -> dict:
        cleaned_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': self.original_shape[0] - cleaned_shape[0],
            'duplicates_removed': self.original_shape[0] - self.df.drop_duplicates().shape[0]
        }

def clean_dataset(df: pd.DataFrame, 
                  remove_dups: bool = True,
                  standardize_cols: bool = True,
                  fill_missing: bool = True) -> pd.DataFrame:
    
    cleaner = DataCleaner(df)
    
    if remove_dups:
        cleaner.remove_duplicates()
    
    if standardize_cols:
        cleaner.standardize_columns()
    
    if fill_missing:
        cleaner.fill_missing_numeric(strategy='mean')
    
    return cleaner.get_cleaned_data()