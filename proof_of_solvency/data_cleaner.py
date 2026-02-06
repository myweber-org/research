import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        for column in cleaned_df.select_dtypes(include=['number']).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': [10, 11, 12, 12, 13]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas Series using the IQR method.
    Returns cleaned Series and outlier indices.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
    cleaned_data = data[mask].copy()
    outlier_indices = data[~mask].index.tolist()
    
    return cleaned_data, outlier_indices

def normalize_minmax(data, column):
    """
    Normalize data to [0,1] range using min-max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns):
    """
    Main cleaning function that processes multiple numeric columns.
    Removes outliers and creates normalized versions.
    """
    cleaned_df = df.copy()
    outlier_report = {}
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            # Remove outliers
            cleaned_df, outliers = remove_outliers_iqr(cleaned_df, col)
            outlier_report[col] = len(outliers)
            
            # Add normalized versions
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
            cleaned_df[f'{col}_standardized'] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df, outlier_report

def validate_dataframe(df, required_columns):
    """
    Validate that dataframe contains required columns and no nulls in key fields.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    null_counts = df[required_columns].isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0].index.tolist()
    
    if columns_with_nulls:
        print(f"Warning: Null values found in columns: {columns_with_nulls}")
    
    return len(columns_with_nulls) == 0import re
import json
from typing import Dict, Any, List, Optional

def clean_string(value: str) -> str:
    """Remove extra whitespace and normalize string."""
    if not isinstance(value, str):
        return str(value)
    return ' '.join(value.strip().split())

def validate_email(email: str) -> bool:
    """Validate email format using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def remove_duplicates(data_list: List[Any]) -> List[Any]:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def sanitize_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize dictionary by cleaning string values."""
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, str):
            sanitized[key] = clean_string(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_json(value)
        elif isinstance(value, list):
            sanitized[key] = [sanitize_json(item) if isinstance(item, dict) else 
                             (clean_string(item) if isinstance(item, str) else item) 
                             for item in value]
        else:
            sanitized[key] = value
    return sanitized

def load_and_validate_config(filepath: str) -> Optional[Dict[str, Any]]:
    """Load and validate JSON configuration file."""
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        if not isinstance(config, dict):
            raise ValueError("Config must be a JSON object")
        
        return sanitize_json(config)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading config: {e}")
        return None

def filter_by_keyword(data_list: List[str], keyword: str) -> List[str]:
    """Filter list of strings containing specific keyword."""
    if not keyword:
        return data_list
    return [item for item in data_list if keyword.lower() in item.lower()]import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        return self
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                df_clean = df_clean[(z_scores < threshold) | df_clean[col].isna()]
        
        self.df = df_clean
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                col_min = df_normalized[col].min()
                col_max = df_normalized[col].max()
                if col_max != col_min:
                    df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        
        self.df = df_normalized
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                col_mean = df_normalized[col].mean()
                col_std = df_normalized[col].std()
                if col_std > 0:
                    df_normalized[col] = (df_normalized[col] - col_mean) / col_std
        
        self.df = df_normalized
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
        
        self.df = df_filled
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        
        self.df = df_filled
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'removed_rows': self.get_removed_count(),
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.iloc[10:20, 0] = np.nan
    df.iloc[100:110, 1] = np.nan
    
    cleaner = DataCleaner(df)
    cleaner.remove_outliers_iqr(['feature1', 'feature2'])
    cleaner.fill_missing_mean(['feature1', 'feature2'])
    cleaner.normalize_minmax(['feature1', 'feature2', 'feature3'])
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Removed rows: {cleaner.get_removed_count()}")
    print(f"Missing values after cleaning: {cleaned_df.isnull().sum().sum()}")
    
    return cleaned_df, summary

if __name__ == "__main__":
    cleaned_data, stats_summary = example_usage()
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}.")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
        print("Filled missing values with mode.")
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.empty:
        print("DataFrame is empty.")
        return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, None, 15.3, 20.1, None, 30.7],
        'category': ['A', 'B', 'B', 'A', 'C', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['id', 'value'])
    print(f"\nData valid: {is_valid}")
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = df.select_dtypes(include=['object']).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].mean()
            )
        elif strategy == 'median' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].median()
            )
        elif strategy == 'mode' and self.categorical_columns.any():
            for col in self.categorical_columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        elif fill_value is not None:
            self.df = self.df.fillna(fill_value)
        return self

    def remove_outliers(self, method='zscore', threshold=3):
        if method == 'zscore' and self.numeric_columns.any():
            z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
            outlier_mask = (z_scores < threshold).all(axis=1)
            self.df = self.df[outlier_mask]
        elif method == 'iqr' and self.numeric_columns.any():
            for col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self

    def get_cleaned_data(self):
        return self.df

    def summary(self):
        print(f"Original shape: {self.df.shape}")
        print(f"Numeric columns: {list(self.numeric_columns)}")
        print(f"Categorical columns: {list(self.categorical_columns)}")
        print(f"Missing values:\n{self.df.isnull().sum()}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_mean(df, column):
    mean_val = df[column].mean()
    df[column] = df[column].fillna(mean_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = handle_missing_mean(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
            cleaned_df = standardize_zscore(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = {'A': [1, 2, 3, 4, 5, 100],
                   'B': [10, 20, None, 40, 50, 60]}
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, ['A', 'B'])
    print("\nCleaned DataFrame:")
    print(cleaned)