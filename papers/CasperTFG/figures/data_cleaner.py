import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns:
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif strategy == 'median' and self.numeric_columns:
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else fill_value, inplace=True)
        elif fill_value is not None:
            self.df.fillna(fill_value, inplace=True)
        return self
    
    def remove_outliers(self, method='zscore', threshold=3):
        if method == 'zscore' and self.numeric_columns:
            z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
            self.df = self.df[(z_scores < threshold).all(axis=1)]
        elif method == 'iqr' and self.numeric_columns:
            for col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self
    
    def normalize_data(self, method='minmax'):
        if method == 'minmax' and self.numeric_columns:
            for col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        elif method == 'standard' and self.numeric_columns:
            for col in self.numeric_columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        return self
    
    def get_cleaned_data(self):
        return self.df

def create_sample_data():
    data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['X', 'Y', 'X', 'Y', np.nan, 'X'],
        'D': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_sample_data()
    cleaner = DataCleaner(df)
    
    cleaned_df = (cleaner
                 .handle_missing_values(strategy='mean')
                 .remove_outliers(method='zscore', threshold=2.5)
                 .normalize_data(method='minmax')
                 .get_cleaned_data())
    
    print("Original data shape:", df.shape)
    print("Cleaned data shape:", cleaned_df.shape)
    print("\nCleaned data:")
    print(cleaned_df)
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset)
        return self
        
    def handle_missing_values(self, 
                             strategy: str = 'drop', 
                             fill_value: Optional[Union[int, float, str]] = None,
                             columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.columns
            
        if strategy == 'drop':
            self.df = self.df.dropna(subset=columns)
        elif strategy == 'fill':
            if fill_value is None:
                raise ValueError("fill_value must be provided when strategy is 'fill'")
            self.df[columns] = self.df[columns].fillna(fill_value)
        elif strategy == 'mean':
            numeric_cols = self.df[columns].select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = self.df[columns].select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            for col in columns:
                if col in self.df.columns:
                    mode_val = self.df[col].mode()
                    if not mode_val.empty:
                        self.df[col] = self.df[col].fillna(mode_val.iloc[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        return self
        
    def remove_outliers_iqr(self, 
                           columns: Optional[List[str]] = None,
                           multiplier: float = 1.5) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                
        return self
        
    def standardize_columns(self, 
                           columns: Optional[List[str]] = None,
                           method: str = 'zscore') -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.number]:
                if method == 'zscore':
                    mean = self.df[col].mean()
                    std = self.df[col].std()
                    if std > 0:
                        self.df[col] = (self.df[col] - mean) / std
                elif method == 'minmax':
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val > min_val:
                        self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                        
        return self
        
    def encode_categorical(self, 
                          columns: Optional[List[str]] = None,
                          method: str = 'onehot') -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns
            
        if method == 'onehot':
            self.df = pd.get_dummies(self.df, columns=columns, drop_first=True)
        elif method == 'label':
            for col in columns:
                if col in self.df.columns:
                    unique_vals = self.df[col].unique()
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    self.df[col] = self.df[col].map(mapping)
                    
        return self
        
    def get_cleaning_report(self) -> Dict:
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': self.df.duplicated().sum()
        }
        
    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy()
        
    def save_to_csv(self, filepath: str, index: bool = False) -> None:
        self.df.to_csv(filepath, index=index)

def load_and_clean_csv(filepath: str, 
                      missing_strategy: str = 'mean',
                      remove_outliers: bool = True) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates()
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if remove_outliers:
        cleaner.remove_outliers_iqr()
        
    report = cleaner.get_cleaning_report()
    print(f"Data cleaning completed:")
    print(f"  Original shape: {report['original_shape']}")
    print(f"  Cleaned shape: {report['cleaned_shape']}")
    print(f"  Rows removed: {report['rows_removed']}")
    print(f"  Missing values remaining: {report['missing_values']}")
    
    return cleaner.get_dataframe()
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {},
        'data_types': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing
    
    validation_result['null_counts'] = df.isnull().sum().to_dict()
    validation_result['data_types'] = df.dtypes.astype(str).to_dict()
    
    return validation_result

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Result:")
    print(validate_dataframe(df))
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)