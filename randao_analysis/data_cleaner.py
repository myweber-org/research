import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.columns
            
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else np.nan, inplace=True)
                elif strategy == 'drop':
                    self.df.dropna(subset=[col], inplace=True)
                else:
                    self.df[col].fillna(method='ffill', inplace=True)
        return self
    
    def convert_dtypes(self, type_mapping: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_mapping.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    elif dtype == 'numeric':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    elif dtype == 'category':
                        self.df[col] = self.df[col].astype('category')
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except (ValueError, TypeError):
                    continue
        return self
    
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataCleaner':
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        return self
    
    def normalize_column(self, column: str, method: str = 'minmax') -> 'DataCleaner':
        if column in self.df.columns and pd.api.types.is_numeric_dtype(self.df[column]):
            if method == 'minmax':
                min_val = self.df[column].min()
                max_val = self.df[column].max()
                if max_val > min_val:
                    self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = self.df[column].mean()
                std_val = self.df[column].std()
                if std_val > 0:
                    self.df[column] = (self.df[column] - mean_val) / std_val
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
    def get_cleaning_report(self) -> Dict:
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict()
        }

def load_and_clean_csv(filepath: str, cleaning_steps: Optional[Dict] = None) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    if cleaning_steps:
        if 'missing_strategy' in cleaning_steps:
            cleaner.handle_missing_values(
                strategy=cleaning_steps['missing_strategy'],
                columns=cleaning_steps.get('missing_columns')
            )
        if 'type_mapping' in cleaning_steps:
            cleaner.convert_dtypes(cleaning_steps['type_mapping'])
        if 'remove_dups' in cleaning_steps and cleaning_steps['remove_dups']:
            cleaner.remove_duplicates(subset=cleaning_steps.get('dup_subset'))
        if 'normalize' in cleaning_steps:
            for norm_config in cleaning_steps['normalize']:
                cleaner.normalize_column(**norm_config)
    
    return cleaner.get_cleaned_data()