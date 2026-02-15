import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'.
                            Default is 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if fill_missing == 'drop':
        df_clean = df_clean.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            elif fill_missing == 'median':
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif fill_missing == 'mode':
        for col in df_clean.columns:
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid, message = validate_data(cleaned_df, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid} - {message}")
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
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
        
        return self
    
    def convert_types(self, type_mapping: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_mapping.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert column {col} to {dtype}: {e}")
        
        return self
    
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df.drop_duplicates(subset=subset, inplace=True, keep='first')
        return self
    
    def normalize_column(self, column: str) -> 'DataCleaner':
        if column in self.df.columns and self.df[column].dtype in [np.float64, np.int64]:
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val > min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        
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

def clean_csv_file(input_path: str, output_path: str, **kwargs) -> Dict:
    try:
        df = pd.read_csv(input_path)
        cleaner = DataCleaner(df)
        
        if 'missing_strategy' in kwargs:
            cleaner.handle_missing_values(strategy=kwargs['missing_strategy'])
        
        if 'type_mapping' in kwargs:
            cleaner.convert_types(kwargs['type_mapping'])
        
        if 'remove_dups' in kwargs and kwargs['remove_dups']:
            cleaner.remove_duplicates()
        
        cleaned_df = cleaner.get_cleaned_data()
        cleaned_df.to_csv(output_path, index=False)
        
        return cleaner.get_cleaning_report()
    
    except Exception as e:
        print(f"Error cleaning file: {e}")
        return {}import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas Series using the IQR method.
    Returns a cleaned Series with outliers removed.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series")
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return data[(data >= lower_bound) & (data <= upper_bound)]

def z_score_normalize(data):
    """
    Normalize data using z-score normalization.
    Returns normalized data with mean=0 and std=1.
    """
    if not isinstance(data, (pd.Series, np.ndarray, list)):
        raise TypeError("Input data must be array-like")
    
    data_array = np.array(data)
    mean_val = np.mean(data_array)
    std_val = np.std(data_array)
    
    if std_val == 0:
        return np.zeros_like(data_array)
    
    return (data_array - mean_val) / std_val

def min_max_normalize(data, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling to specified range.
    Default range is [0, 1].
    """
    if not isinstance(data, (pd.Series, np.ndarray, list)):
        raise TypeError("Input data must be array-like")
    
    data_array = np.array(data)
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    
    if max_val == min_val:
        return np.full_like(data_array, feature_range[0])
    
    normalized = (data_array - min_val) / (max_val - min_val)
    scaled = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return scaled

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method=None):
    """
    Clean a pandas DataFrame by removing outliers and optionally normalizing numeric columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = df.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        original_data = cleaned_df[col].dropna()
        
        if outlier_method == 'iqr':
            mask = ~cleaned_df[col].isna()
            cleaned_series = remove_outliers_iqr(original_data, col)
            cleaned_df.loc[mask, col] = cleaned_series.reindex(cleaned_df[mask].index)
        
        if normalize_method == 'zscore':
            cleaned_df[col] = z_score_normalize(cleaned_df[col])
        elif normalize_method == 'minmax':
            cleaned_df[col] = min_max_normalize(cleaned_df[col])
    
    return cleaned_df

def validate_data(data, check_missing=True, check_infinite=True):
    """
    Validate data for common issues like missing values and infinite values.
    Returns a dictionary with validation results.
    """
    validation_results = {}
    
    if isinstance(data, pd.DataFrame):
        if check_missing:
            missing_counts = data.isnull().sum()
            validation_results['missing_counts'] = missing_counts.to_dict()
            validation_results['total_missing'] = missing_counts.sum()
        
        if check_infinite:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            inf_counts = {}
            for col in numeric_cols:
                inf_mask = np.isinf(data[col])
                inf_counts[col] = inf_mask.sum()
            validation_results['infinite_counts'] = inf_counts
    
    return validation_results