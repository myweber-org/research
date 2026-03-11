
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
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        if columns is None:
            columns = self.df.columns
        
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean' and np.issubdtype(self.df[col].dtype, np.number):
                    fill_value = self.df[col].mean()
                elif strategy == 'median' and np.issubdtype(self.df[col].dtype, np.number):
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    continue
                else:
                    fill_value = 0
                
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in column '{col}' using {strategy} strategy")
    
    def remove_duplicates(self, subset=None, keep='first'):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = initial_rows - len(self.df)
        print(f"Removed {removed} duplicate rows")
    
    def normalize_column(self, column, method='minmax'):
        if self.df is None or column not in self.df.columns:
            print(f"Column '{column}' not found in data")
            return
        
        if not np.issubdtype(self.df[column].dtype, np.number):
            print(f"Column '{column}' is not numeric")
            return
        
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val != 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
        
        print(f"Normalized column '{column}' using {method} method")
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            print("No data to save")
            return
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to: {output_path}")
        return output_path
    
    def get_summary(self):
        if self.df is None:
            return {}
        
        summary = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def clean_csv_file(input_file, output_file=None, missing_strategy='mean'):
    cleaner = DataCleaner(input_file)
    
    if not cleaner.load_data():
        return None
    
    print("Initial data summary:", cleaner.get_summary())
    
    cleaner.handle_missing_values(strategy=missing_strategy)
    cleaner.remove_duplicates()
    
    numeric_cols = cleaner.df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaner.normalize_column(col)
    
    output_path = cleaner.save_cleaned_data(output_file)
    print("Final data summary:", cleaner.get_summary())
    
    return output_path

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'category': ['A', 'B', 'A', None, 'C', 'A']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = Path('test_data.csv')
    test_df.to_csv(test_file, index=False)
    
    result = clean_csv_file('test_data.csv', 'cleaned_test_data.csv')
    
    if result:
        print(f"Data cleaning completed. Output saved to: {result}")
    
    test_file.unlink()
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers from a column using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize a column using Min-Max scaling to range [0, 1].
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized column values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    col_data = data[column]
    min_val = col_data.min()
    max_val = col_data.max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(col_data), index=col_data.index)
    
    normalized = (col_data - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize a column using Z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.Series: Standardized column values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    col_data = data[column]
    mean_val = col_data.mean()
    std_val = col_data.std()
    
    if std_val == 0:
        return pd.Series([0] * len(col_data), index=col_data.index)
    
    standardized = (col_data - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5, normalize=True):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    outlier_multiplier (float): IQR multiplier for outlier removal
    normalize (bool): Whether to normalize numeric columns
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_multiplier)
            
            if normalize:
                cleaned_data[f"{column}_normalized"] = normalize_minmax(cleaned_data, column)
                cleaned_data[f"{column}_standardized"] = standardize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns, allow_nan=False):
    """
    Validate dataframe structure and content.
    
    Parameters:
    data (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    allow_nan (bool): Whether NaN values are allowed
    
    Returns:
    tuple: (is_valid, error_message)
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    if not allow_nan:
        nan_columns = data.columns[data.isnull().any()].tolist()
        if nan_columns:
            return False, f"NaN values found in columns: {nan_columns}"
    
    return True, "Data validation passed"

def main():
    sample_data = pd.DataFrame({
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'score': np.random.uniform(0, 100, 100)
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original data statistics:")
    print(sample_data.describe())
    
    cleaned = clean_dataset(sample_data, numeric_columns=['value', 'score'])
    print("\nCleaned data shape:", cleaned.shape)
    print("Cleaned data statistics:")
    print(cleaned.describe())
    
    is_valid, message = validate_data(cleaned, ['id', 'value', 'score'])
    print(f"\nData validation: {message}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df: pd.DataFrame, 
                         column_types: dict) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
    Args:
        df: Input DataFrame
        column_types: Dictionary mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_converted = df.copy()
    
    for column, dtype in column_types.items():
        if column in df_converted.columns:
            try:
                df_converted[column] = df_converted[column].astype(dtype)
            except (ValueError, TypeError):
                df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
    
    return df_converted

def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'mean',
                         columns: List[str] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: Specific columns to process
    
    Returns:
        DataFrame with handled missing values
    """
    df_processed = df.copy()
    
    if columns is None:
        columns = df_processed.columns
    
    for column in columns:
        if column in df_processed.columns:
            if strategy == 'drop':
                df_processed = df_processed.dropna(subset=[column])
            elif strategy == 'mean':
                df_processed[column] = df_processed[column].fillna(df_processed[column].mean())
            elif strategy == 'median':
                df_processed[column] = df_processed[column].fillna(df_processed[column].median())
            elif strategy == 'mode':
                df_processed[column] = df_processed[column].fillna(df_processed[column].mode()[0])
    
    return df_processed

def normalize_column(df: pd.DataFrame, 
                    column: str,
                    method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize a column using specified method.
    
    Args:
        df: Input DataFrame
        column: Column to normalize
        method: Normalization method ('minmax', 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    df_normalized = df.copy()
    
    if column not in df_normalized.columns:
        return df_normalized
    
    if method == 'minmax':
        col_min = df_normalized[column].min()
        col_max = df_normalized[column].max()
        if col_max != col_min:
            df_normalized[column] = (df_normalized[column] - col_min) / (col_max - col_min)
    
    elif method == 'zscore':
        col_mean = df_normalized[column].mean()
        col_std = df_normalized[column].std()
        if col_std != 0:
            df_normalized[column] = (df_normalized[column] - col_mean) / col_std
    
    return df_normalized

def clean_dataframe(df: pd.DataFrame,
                   duplicate_subset: List[str] = None,
                   type_conversions: dict = None,
                   missing_strategy: str = 'mean',
                   normalize_columns: List[str] = None) -> pd.DataFrame:
    """
    Apply comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        duplicate_subset: Columns for duplicate removal
        type_conversions: Column type conversions
        missing_strategy: Strategy for handling missing values
        normalize_columns: Columns to normalize
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    cleaned_df = remove_duplicates(cleaned_df, duplicate_subset)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_columns:
        for column in normalize_columns:
            cleaned_df = normalize_column(cleaned_df, column)
    
    return cleaned_dfimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    for col in numeric_columns:
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, numeric_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    for col in numeric_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df