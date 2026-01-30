
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.columns
            
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean' and self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median' and self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 0, inplace=True)
                elif strategy == 'drop':
                    self.df.dropna(subset=[col], inplace=True)
                else:
                    self.df[col].fillna(0, inplace=True)
        return self
        
    def convert_types(self, type_map: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_map.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                    elif dtype == 'numeric':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    elif dtype == 'category':
                        self.df[col] = self.df[col].astype('category')
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except Exception as e:
                    print(f"Error converting column {col}: {e}")
        return self
        
    def remove_outliers(self, column: str, method: str = 'iqr', threshold: float = 1.5) -> 'DataCleaner':
        if column not in self.df.columns:
            return self
            
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.df[column]))
            self.df = self.df[z_scores < threshold]
            
        return self
        
    def normalize_column(self, column: str, method: str = 'minmax') -> 'DataCleaner':
        if column not in self.df.columns:
            return self
            
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val > min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
                
        elif method == 'standard':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
                
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> Dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values_remaining': self.df.isnull().sum().sum()
        }

def clean_csv_file(filepath: str, output_path: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    cleaner.handle_missing_values(strategy='mean')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaner.remove_outliers(col, method='iqr')
        
    report = cleaner.get_cleaning_report()
    print(f"Cleaning complete. Removed {report['rows_removed']} rows.")
    
    cleaned_df = cleaner.get_cleaned_data()
    
    if output_path:
        cleaned_df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    
    return cleaned_df
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing statistics
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'count': data[column].count()
    }
    return stats

def normalize_column(data, column):
    """
    Normalize a column using min-max scaling.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val != min_val:
        data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    else:
        data[column + '_normalized'] = 0
    
    return dataimport pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean', remove_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_check (list): List of columns to check for missing values. If None, checks all columns.
    fill_missing (str): Method to fill missing values - 'mean', 'median', 'mode', or 'drop'
    remove_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    if columns_to_check is None:
        columns_to_check = df_clean.columns.tolist()
    
    if fill_missing != 'drop':
        for col in columns_to_check:
            if df_clean[col].isnull().any():
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif fill_missing == 'mode':
                    df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0, inplace=True)
                else:
                    df_clean[col].fillna(0, inplace=True)
    else:
        df_clean.dropna(subset=columns_to_check, inplace=True)
    
    if remove_duplicates:
        df_clean.drop_duplicates(inplace=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def get_data_summary(df):
    """
    Generate a summary of the DataFrame including missing values and data types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Summary DataFrame
    """
    
    summary_data = []
    
    for col in df.columns:
        col_info = {
            'column': col,
            'dtype': str(df[col].dtype),
            'non_null_count': df[col].count(),
            'null_count': df[col].isnull().sum(),
            'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
            'unique_values': df[col].nunique()
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info['mean'] = df[col].mean()
            col_info['std'] = df[col].std()
            col_info['min'] = df[col].min()
            col_info['max'] = df[col].max()
        
        summary_data.append(col_info)
    
    return pd.DataFrame(summary_data)