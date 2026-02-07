
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        
        removed_count = len(self.df) - len(clean_df)
        self.df = clean_df
        return removed_count
    
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in normalized_df.columns:
                if method == 'zscore':
                    normalized_df[col] = stats.zscore(normalized_df[col])
                elif method == 'minmax':
                    min_val = normalized_df[col].min()
                    max_val = normalized_df[col].max()
                    if max_val != min_val:
                        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                elif method == 'robust':
                    median = normalized_df[col].median()
                    iqr = normalized_df[col].quantile(0.75) - normalized_df[col].quantile(0.25)
                    if iqr != 0:
                        normalized_df[col] = (normalized_df[col] - median) / iqr
        
        self.df = normalized_df
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in filled_df.columns and filled_df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = filled_df[col].mean()
                elif strategy == 'median':
                    fill_value = filled_df[col].median()
                elif strategy == 'mode':
                    fill_value = filled_df[col].mode()[0]
                elif strategy == 'drop':
                    filled_df = filled_df.dropna(subset=[col])
                    continue
                else:
                    fill_value = 0
                
                filled_df[col] = filled_df[col].fillna(fill_value)
        
        self.df = filled_df
        return self.df
    
    def get_cleaned_data(self):
        return self.df
    
    def get_cleaning_stats(self):
        return {
            'original_rows': self.original_shape[0],
            'cleaned_rows': len(self.df),
            'rows_removed': self.original_shape[0] - len(self.df),
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1]
        }

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    cleaner.handle_missing_values(strategy='median')
    cleaner.remove_outliers_iqr(factor=1.5)
    cleaner.normalize_data(method='zscore')
    
    stats = cleaner.get_cleaning_stats()
    cleaned_df = cleaner.get_cleaned_data()
    
    return cleaned_df, statsimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method=None):
    """
    Main function to clean dataset with optional outlier removal and normalization
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Remove outliers
    if outlier_method == 'iqr':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
    elif outlier_method == 'zscore':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    # Apply normalization
    if normalize_method == 'minmax':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
    elif normalize_method == 'zscore':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate summary statistics for the dataset
    """
    summary = {
        'original_rows': len(data),
        'original_columns': len(data.columns),
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': data.isnull().sum().to_dict(),
        'data_types': data.dtypes.to_dict()
    }
    
    for col in summary['numeric_columns']:
        summary[f'{col}_stats'] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'median': data[col].median()
        }
    
    return summary
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default True.
    fill_missing (str): Method to fill missing values. 
                        Options: 'mean', 'median', 'mode', or 'drop'. Default 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            elif fill_missing == 'mode':
                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None],
        'B': [4, None, 6, 6, 8],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {message}")
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset)
        return self
        
    def handle_missing_values(self, strategy: str = 'drop', fill_value: Optional[float] = None) -> 'DataCleaner':
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                self.df = self.df.fillna(self.df.mean())
        return self
        
    def remove_outliers(self, column: str, threshold: float = 3.0) -> 'DataCleaner':
        if column in self.df.columns:
            z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
            self.df = self.df[z_scores < threshold]
        return self
        
    def normalize_column(self, column: str) -> 'DataCleaner':
        if column in self.df.columns and self.df[column].std() != 0:
            self.df[column] = (self.df[column] - self.df[column].mean()) / self.df[column].std()
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_summary(self) -> dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1]
        }

def clean_dataset(df: pd.DataFrame, steps: List[dict]) -> pd.DataFrame:
    cleaner = DataCleaner(df)
    
    for step in steps:
        action = step.get('action')
        
        if action == 'remove_duplicates':
            subset = step.get('subset')
            cleaner.remove_duplicates(subset)
            
        elif action == 'handle_missing':
            strategy = step.get('strategy', 'drop')
            fill_value = step.get('fill_value')
            cleaner.handle_missing_values(strategy, fill_value)
            
        elif action == 'remove_outliers':
            column = step.get('column')
            threshold = step.get('threshold', 3.0)
            cleaner.remove_outliers(column, threshold)
            
        elif action == 'normalize':
            column = step.get('column')
            cleaner.normalize_column(column)
    
    return cleaner.get_cleaned_data()