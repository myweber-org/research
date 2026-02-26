
import numpy as np
import pandas as pd

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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
            cleaned_df = standardize_zscore(cleaned_df, col)
    return cleaned_dfimport numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val - min_val == 0:
        return data[column]
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='standard'):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
        if outlier_removal:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        if normalization == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalization == 'standard':
            cleaned_df[col] = standardize_zscore(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
        
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
        return self
        
    def fill_missing_mean(self, column):
        self.df[column].fillna(self.df[column].mean(), inplace=True)
        return self
        
    def fill_missing_median(self, column):
        self.df[column].fillna(self.df[column].median(), inplace=True)
        return self
        
    def fill_missing_mode(self, column):
        self.df[column].fillna(self.df[column].mode()[0], inplace=True)
        return self
        
    def drop_missing_rows(self, threshold=0.8):
        self.df.dropna(thresh=threshold * len(self.df.columns), inplace=True)
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
        
    def summary(self):
        print(f"Original rows: {self.original_shape[0]}")
        print(f"Cleaned rows: {self.df.shape[0]}")
        print(f"Rows removed: {self.get_removed_count()}")
        print(f"Original columns: {self.original_shape[1]}")
        print(f"Cleaned columns: {self.df.shape[1]}")

def clean_dataset(df, config):
    cleaner = DataCleaner(df)
    
    if 'outlier_method' in config:
        method = config['outlier_method']
        columns = config.get('outlier_columns', df.select_dtypes(include=[np.number]).columns)
        
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if method == 'iqr':
                    cleaner.remove_outliers_iqr(col, config.get('iqr_multiplier', 1.5))
                elif method == 'zscore':
                    cleaner.remove_outliers_zscore(col, config.get('zscore_threshold', 3))
    
    if 'missing_strategy' in config:
        strategy = config['missing_strategy']
        columns = config.get('missing_columns', df.columns)
        
        for col in columns:
            if col in df.columns and self.df[col].isnull().any():
                if strategy == 'mean':
                    cleaner.fill_missing_mean(col)
                elif strategy == 'median':
                    cleaner.fill_missing_median(col)
                elif strategy == 'mode':
                    cleaner.fill_missing_mode(col)
                elif strategy == 'drop':
                    cleaner.drop_missing_rows(config.get('drop_threshold', 0.8))
    
    return cleaner.get_cleaned_data()
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe: Input DataFrame
        subset: Column label or sequence of labels to consider for duplicates
        keep: Determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if dataframe.empty:
        return dataframe
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        dataframe: Input DataFrame
        columns: List of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for column in columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')
    
    return dataframe

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: Input DataFrame
        required_columns: List of required column names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if dataframe.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    mask = z_scores < threshold
    return data[mask]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame or Series
        column: column name to normalize
    
    Returns:
        Normalized data
    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column]
    else:
        series = data
    
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return series * 0
    
    return (series - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame or Series
        column: column name to normalize
    
    Returns:
        Standardized data
    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column]
    else:
        series = data
    
    mean_val = series.mean()
    std_val = series.std()
    
    if std_val == 0:
        return series * 0
    
    return (series - mean_val) / std_val

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric columns to process (default: all numeric)
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    # Remove outliers
    if outlier_method == 'iqr':
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    elif outlier_method == 'zscore':
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    # Normalize data
    if normalize_method == 'minmax':
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = normalize_minmax(cleaned_df, col)
    elif normalize_method == 'zscore':
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns=None, allow_nan=True, max_nan_ratio=0.1):
    """
    Validate data quality.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required columns
        allow_nan: Whether NaN values are allowed
        max_nan_ratio: Maximum allowed ratio of NaN values per column
    
    Returns:
        Tuple of (is_valid, issues_list)
    """
    issues = []
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
    
    # Check NaN values
    if not allow_nan:
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            issues.append(f"Columns with NaN values: {nan_columns}")
    else:
        for col in df.columns:
            nan_ratio = df[col].isna().mean()
            if nan_ratio > max_nan_ratio:
                issues.append(f"Column '{col}' has {nan_ratio:.1%} NaN values (max allowed: {max_nan_ratio:.1%})")
    
    # Check numeric columns for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            issues.append(f"Column '{col}' contains infinite values")
    
    is_valid = len(issues) == 0
    return is_valid, issues
import csv
import hashlib
from collections import defaultdict

def generate_row_hash(row):
    """Generate a hash for a CSV row to identify duplicates."""
    row_string = ''.join(str(field) for field in row)
    return hashlib.md5(row_string.encode()).hexdigest()

def remove_duplicates(input_file, output_file, key_columns=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        key_columns (list): List of column indices to consider for duplicates.
                          If None, entire row is considered.
    """
    seen_hashes = set()
    unique_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        
        for row in reader:
            if key_columns:
                key_data = [row[i] for i in key_columns]
                row_hash = generate_row_hash(key_data)
            else:
                row_hash = generate_row_hash(row)
            
            if row_hash not in seen_hashes:
                seen_hashes.add(row_hash)
                unique_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(unique_rows)
    
    print(f"Removed {len(seen_hashes) - len(unique_rows)} duplicate rows")
    print(f"Original: {len(seen_hashes)} rows, Cleaned: {len(unique_rows)} rows")

def find_duplicate_stats(input_file, key_columns=None):
    """
    Analyze duplicate patterns in CSV data.
    
    Returns:
        dict: Statistics about duplicates
    """
    hash_counts = defaultdict(int)
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header
        
        for row in reader:
            if key_columns:
                key_data = [row[i] for i in key_columns]
                row_hash = generate_row_hash(key_data)
            else:
                row_hash = generate_row_hash(row)
            
            hash_counts[row_hash] += 1
    
    duplicate_hashes = {h: c for h, c in hash_counts.items() if c > 1}
    
    stats = {
        'total_rows': sum(hash_counts.values()),
        'unique_rows': len(hash_counts),
        'duplicate_rows': sum(hash_counts.values()) - len(hash_counts),
        'duplicate_groups': len(duplicate_hashes),
        'max_duplicates': max(hash_counts.values()) if hash_counts else 0
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    input_csv = "data.csv"
    output_csv = "cleaned_data.csv"
    
    # Remove duplicates based on first 3 columns
    remove_duplicates(input_csv, output_csv, key_columns=[0, 1, 2])
    
    # Get statistics
    stats = find_duplicate_stats(input_csv, key_columns=[0, 1, 2])
    for key, value in stats.items():
        print(f"{key}: {value}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
        else:
            df_copy[f'{column}_normalized'] = 0.5
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

if __name__ == "__main__":
    # Example usage
    sample_data = {'values': [10, 12, 12, 13, 12, 11, 100, 14, 12, 10, 9, 15, 200]}
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\nSummary Statistics:")
    print(calculate_summary_statistics(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame (outliers removed):")
    print(cleaned_df)
    
    normalized_df = normalize_column(df, 'values', method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized_df)
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

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

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_filled = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_filled[col] = data[col].fillna(fill_value)
    
    return data_filled

def validate_dataframe(data, required_columns=None):
    """
    Validate DataFrame structure and content
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (bool): Whether to fill missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna('Unknown')
        
        print(f"Filled missing values in {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns")
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame for common data quality issues.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    
    Returns:
    dict: Dictionary containing validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_results

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    
    Parameters:
    df (pd.DataFrame): DataFrame to save
    output_path (str): Path to save the cleaned data
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, None]
    })
    
    print("Original data:")
    print(sample_data)
    print("\nValidation results:")
    print(validate_dataframe(sample_data))
    
    cleaned = clean_dataframe(sample_data)
    print("\nCleaned data:")
    print(cleaned)
    print("\nValidation results after cleaning:")
    print(validate_dataframe(cleaned))
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(exclude=[np.number]).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mean())
        elif strategy == 'median':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].median())
        elif strategy == 'mode':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mode().iloc[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(fill_value)
        else:
            raise ValueError("Invalid strategy or missing fill_value for constant strategy")
        
        self.df[self.categorical_columns] = self.df[self.categorical_columns].fillna('Unknown')
        return self

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self

    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
        return self

    def normalize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        return self

    def get_cleaned_data(self):
        return self.df

    def summary(self):
        print("Data shape:", self.df.shape)
        print("\nMissing values after cleaning:")
        print(self.df.isnull().sum())
        print("\nData types:")
        print(self.df.dtypes)
        print("\nBasic statistics:")
        print(self.df.describe())