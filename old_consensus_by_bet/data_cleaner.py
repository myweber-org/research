
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
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
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}.")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
        print("Filled missing values with mode.")
    
    return cleaned_df

def validate_dataframe(df, check_missing=True, check_types=True):
    """
    Validate DataFrame for common data quality issues.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    check_missing (bool): Check for missing values.
    check_types (bool): Check for consistent data types.
    
    Returns:
    dict: Dictionary containing validation results.
    """
    validation_results = {}
    
    if check_missing:
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        validation_results['missing_values'] = missing_cols.to_dict()
    
    if check_types:
        type_info = {}
        for col in df.columns:
            unique_types = df[col].apply(type).unique()
            type_info[col] = [str(t) for t in unique_types]
        validation_results['data_types'] = type_info
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None, 6],
        'B': [10, None, 30, 40, 50, 60],
        'C': ['x', 'y', 'y', 'z', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataframe(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    validation = validate_dataframe(cleaned)
    print("Validation Results:")
    print(validation)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str or dict): Method to fill missing values. 
                                Can be 'mean', 'median', 'mode', or a dictionary of column:value pairs.
                                If None, missing values are not filled.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
                else:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
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
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def z_score_normalize(data, column):
    """
    Normalize data using z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5, normalization_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    stats_report = {}
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        original_count = len(cleaned_df)
        cleaned_df, outliers_removed = remove_outliers_iqr(cleaned_df, col, outlier_factor)
        stats_report[col] = {
            'original_samples': original_count,
            'outliers_removed': outliers_removed,
            'remaining_samples': len(cleaned_df)
        }
        
        if normalization_method == 'minmax':
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
        elif normalization_method == 'zscore':
            cleaned_df[f'{col}_normalized'] = z_score_normalize(cleaned_df, col)
    
    return cleaned_df, stats_report

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {},
        'data_types': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['is_valid'] = False
            validation_results['missing_columns'] = missing
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        validation_results['null_counts'][col] = null_count
        validation_results['data_types'][col] = str(df[col].dtype)
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[np.random.choice(1000, 50), 'feature_a'] = np.random.uniform(500, 1000, 50)
    
    validation = validate_dataframe(df, ['feature_a', 'feature_b', 'category'])
    print("Validation Results:", validation)
    
    cleaned_df, report = clean_dataset(df, ['feature_a', 'feature_b'])
    print("\nCleaning Report:", report)
    print(f"\nOriginal shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")import csv
import re
from typing import List, Dict, Optional

def clean_string(value: str) -> str:
    """Remove extra whitespace and normalize string."""
    if not isinstance(value, str):
        return str(value)
    cleaned = re.sub(r'\s+', ' ', value.strip())
    return cleaned

def clean_numeric(value: str) -> Optional[float]:
    """Convert string to float, handling common issues."""
    if not value:
        return None
    cleaned = value.replace(',', '').replace('$', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return None

def clean_csv_row(row: Dict[str, str]) -> Dict[str, str]:
    """Apply cleaning functions to all fields in a row."""
    cleaned_row = {}
    for key, value in row.items():
        if any(num_key in key.lower() for num_key in ['price', 'amount', 'quantity', 'total']):
            cleaned_value = clean_numeric(value)
            cleaned_row[key] = str(cleaned_value) if cleaned_value is not None else ''
        else:
            cleaned_row[key] = clean_string(value)
    return cleaned_row

def process_csv_file(input_path: str, output_path: str) -> None:
    """Read CSV, clean data, and write to new file."""
    with open(input_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                cleaned_row = clean_csv_row(row)
                writer.writerow(cleaned_row)

def validate_email(email: str) -> bool:
    """Basic email validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def remove_duplicates(data: List[Dict], key_field: str) -> List[Dict]:
    """Remove duplicate entries based on specified key field."""
    seen = set()
    unique_data = []
    for item in data:
        key = item.get(key_field)
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    return unique_data
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        
    def remove_outliers_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
        
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
        return self
        
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            self.df[column] = (self.df[column] - mean_val) / std_val
        return self
        
    def fill_missing(self, column, method='mean'):
        if method == 'mean':
            fill_value = self.df[column].mean()
        elif method == 'median':
            fill_value = self.df[column].median()
        elif method == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = method
            
        self.df[column].fillna(fill_value, inplace=True)
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def reset_to_original(self):
        self.df = self.original_df.copy()
        return self
        
    def summary(self):
        print("Data Cleaning Summary")
        print(f"Original rows: {len(self.original_df)}")
        print(f"Cleaned rows: {len(self.df)}")
        print(f"Rows removed: {len(self.original_df) - len(self.df)}")
        print(f"Columns: {list(self.df.columns)}")
        return self

def clean_dataset(df, config):
    cleaner = DataCleaner(df)
    
    for column, operations in config.items():
        if 'remove_outliers' in operations:
            method = operations['remove_outliers']
            if method == 'iqr':
                cleaner.remove_outliers_iqr(column)
            elif method == 'zscore':
                threshold = operations.get('threshold', 3)
                cleaner.remove_outliers_zscore(column, threshold)
                
        if 'normalize' in operations:
            method = operations['normalize']
            cleaner.normalize_column(column, method)
            
        if 'fill_missing' in operations:
            method = operations['fill_missing']
            cleaner.fill_missing(column, method)
    
    return cleaner.get_cleaned_data()
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    factor (float): Multiplier for IQR
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.Series: Standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to process, None for all columns
    
    Returns:
    pd.DataFrame: Dataframe with handled missing values
    """
    if columns is None:
        columns = data.columns
    
    result = data.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if result[col].isnull().any():
            if strategy == 'mean':
                fill_value = result[col].mean()
            elif strategy == 'median':
                fill_value = result[col].median()
            elif strategy == 'mode':
                fill_value = result[col].mode()[0] if not result[col].mode().empty else 0
            elif strategy == 'drop':
                result = result.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            result[col] = result[col].fillna(fill_value)
    
    return result

def create_sample_data():
    """
    Create sample data for testing.
    
    Returns:
    pd.DataFrame: Sample dataframe with test data
    """
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.uniform(0, 1, 100),
        'feature_c': np.random.exponential(2, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(df.index, size=10, replace=False), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, size=5, replace=False), 'feature_b'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    print("\nMissing values per column:")
    print(sample_data.isnull().sum())
    
    cleaned_data = handle_missing_values(sample_data, strategy='mean')
    print("\nAfter handling missing values:")
    print("Data shape:", cleaned_data.shape)
    print("Missing values:", cleaned_data.isnull().sum().sum())
    
    filtered_data = remove_outliers_iqr(cleaned_data, 'feature_a')
    print("\nAfter outlier removal:")
    print("Data shape:", filtered_data.shape)
    
    normalized_values = normalize_minmax(filtered_data, 'feature_b')
    print("\nNormalized feature_b (first 5 values):")
    print(normalized_values.head())
    
    standardized_values = standardize_zscore(filtered_data, 'feature_c')
    print("\nStandardized feature_c (first 5 values):")
    print(standardized_values.head())import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing statistical measures.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 102, 12, 14, 13, 12, 11, 14, 13, 12, 14, 100]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nOriginal Statistics:")
    print(calculate_basic_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Statistics:")
    print(calculate_basic_stats(cleaned_df, 'values'))import pandas as pd

def clean_dataframe(df, column_names=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_names (list, optional): List of column names to normalize. 
                                      If None, all object dtype columns are normalized.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Determine which columns to normalize
    if column_names is None:
        # Get all object dtype columns (typically strings)
        column_names = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    # Normalize string columns: strip whitespace and convert to lowercase
    for col in column_names:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     sample_data = {
#         'name': ['  John  ', 'Jane', 'JOHN', '  Jane  ', 'Bob'],
#         'age': [25, 30, 25, 30, 35],
#         'city': ['New York', '  los angeles  ', 'NEW YORK', 'Los Angeles', 'Chicago']
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     print(f"\nShape: {df.shape}")
#     
#     # Clean the data
#     cleaned = clean_dataframe(df)
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     print(f"\nShape: {cleaned.shape}")
#     
#     # Validate the cleaned data
#     is_valid, message = validate_dataframe(cleaned, ['name', 'age', 'city'])
#     print(f"\nValidation: {is_valid} - {message}")