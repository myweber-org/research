
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Remove outliers using z-score
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fill_method (str or None): Method to fill missing values: 
                               'mean', 'median', 'mode', or None to drop rows.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method is None:
        cleaned_df = cleaned_df.dropna()
    else:
        if fill_method == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_method == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_method == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        else:
            raise ValueError("fill_method must be 'mean', 'median', 'mode', or None")
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_cols
        validation_results['all_required_columns_present'] = len(missing_cols) == 0
    
    return validation_results

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'A': [1, 2, None, 4, 2],
#         'B': [5, None, 7, 8, 5],
#         'C': ['x', 'y', 'z', 'x', 'y']
#     }
#     df = pd.DataFrame(sample_data)
#     
#     print("Original dataset:")
#     print(df)
#     print("\nValidation results:")
#     print(validate_dataset(df))
#     
#     cleaned = clean_dataset(df, remove_duplicates=True, fill_method='mean')
#     print("\nCleaned dataset:")
#     print(cleaned)
#     print("\nCleaned validation results:")
#     print(validate_dataset(cleaned))
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_dataimport pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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
    
    return filtered_df.reset_index(drop=True)

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistical measures
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[np.random.choice(df.index, 50), 'A'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("\nColumn A statistics before cleaning:")
    print(calculate_basic_stats(df, 'A'))
    
    cleaned_df = clean_numeric_data(df, ['A'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nColumn A statistics after cleaning:")
    print(calculate_basic_stats(cleaned_df, 'A'))
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
    Calculate summary statistics for a column after outlier removal.
    
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
        'count': df[column].count()
    }
    
    return stats

def process_numeric_columns(df, columns=None):
    """
    Process multiple numeric columns to remove outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, processes all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from specified columns
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    result_df = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            result_df = remove_outliers_iqr(result_df, col)
    
    return result_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[::100, 'A'] = 500
    
    print("Original DataFrame shape:", sample_df.shape)
    print("Original summary for column A:")
    print(calculate_summary_statistics(sample_df, 'A'))
    
    cleaned_df = process_numeric_columns(sample_df, ['A'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary for column A:")
    print(calculate_summary_statistics(cleaned_df, 'A'))import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        removed_count = len(self.df) - len(df_clean)
        self.df = df_clean
        return removed_count
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                df_clean = df_clean[(z_scores < threshold) | df_clean[col].isna()]
        
        removed_count = len(self.df) - len(df_clean)
        self.df = df_clean
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        return self.df
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        return self.df
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        
        return self.df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        return self.df
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removal_stats(self):
        final_shape = self.df.shape
        rows_removed = self.original_shape[0] - final_shape[0]
        cols_removed = self.original_shape[1] - final_shape[1]
        return {
            'original_rows': self.original_shape[0],
            'current_rows': final_shape[0],
            'rows_removed': rows_removed,
            'original_cols': self.original_shape[1],
            'current_cols': final_shape[1],
            'cols_removed': cols_removed
        }

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.random.normal(300, 50, 50)
    df.loc[np.random.choice(df.index, 30), 'feature_b'] = np.random.exponential(200, 30)
    df.loc[np.random.choice(df.index, 100), 'feature_c'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    print("Missing values:\n", sample_df.isnull().sum())
    
    cleaner = DataCleaner(sample_df)
    outliers_removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"Removed {outliers_removed} outliers using IQR method")
    
    cleaner.fill_missing_mean(['feature_c'])
    cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    
    cleaned_df = cleaner.get_cleaned_data()
    print("Cleaned data shape:", cleaned_df.shape)
    print("Missing values after cleaning:\n", cleaned_df.isnull().sum())
    
    stats = cleaner.get_removal_stats()
    print("\nCleaning statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing specified string columns.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = cleaned_df.shape[0]
    cleaned_df.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows - cleaned_df.shape[0]
    
    # Normalize string columns
    if columns_to_clean is None:
        # Automatically detect string columns
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            cleaned_df[column] = cleaned_df[column].apply(normalize_string)
    
    return cleaned_df, removed_duplicates

def normalize_string(value):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(value):
        return value
    
    if isinstance(value, str):
        # Convert to lowercase
        normalized = value.lower()
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        # Remove special characters except alphanumeric and spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized
    
    return value

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and non-null values.
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    # Check for null values in each column
    for column in df.columns:
        null_count = df[column].isnull().sum()
        if null_count > 0:
            validation_results['null_counts'][column] = null_count
    
    return validation_results

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'name': ['John Doe', 'JANE SMITH', 'John Doe', 'Bob   Wilson!'],
#         'email': ['john@example.com', 'jane@example.com', 'john@example.com', 'bob@example.com'],
#         'age': [25, 30, 25, 35]
#     }
#     
#     df = pd.DataFrame(sample_data)
#     cleaned_df, duplicates_removed = clean_dataframe(df)
#     print(f"Removed {duplicates_removed} duplicate rows")
#     print(cleaned_df)
#     
#     validation = validate_dataframe(cleaned_df, required_columns=['name', 'email'])
#     print(f"Data validation: {validation}")
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_data(df, columns, method='minmax'):
    normalized_df = df.copy()
    for col in columns:
        if method == 'minmax':
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            if std_val > 0:
                normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    return normalized_df

def clean_dataset(file_path, numeric_columns):
    try:
        df = pd.read_csv(file_path)
        print(f"Original dataset shape: {df.shape}")
        
        df_cleaned = remove_outliers_iqr(df, numeric_columns)
        print(f"After outlier removal: {df_cleaned.shape}")
        
        df_normalized = normalize_data(df_cleaned, numeric_columns, method='zscore')
        print("Data cleaning and normalization completed.")
        
        return df_normalized
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'feature3': np.random.uniform(0, 1, 200)
    })
    
    sample_data.to_csv('sample_dataset.csv', index=False)
    
    cleaned_data = clean_dataset('sample_dataset.csv', ['feature1', 'feature2', 'feature3'])
    
    if cleaned_data is not None:
        print(cleaned_data.describe())
        cleaned_data.to_csv('cleaned_dataset.csv', index=False)
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(dataframe, columns):
    cleaned_df = dataframe.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_data(dataframe, columns, method='minmax'):
    normalized_df = dataframe.copy()
    for col in columns:
        if col in normalized_df.columns:
            if method == 'minmax':
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val != 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    return normalized_df

def handle_missing_values(dataframe, columns, strategy='mean'):
    processed_df = dataframe.copy()
    for col in columns:
        if col in processed_df.columns:
            if strategy == 'mean':
                fill_value = processed_df[col].mean()
            elif strategy == 'median':
                fill_value = processed_df[col].median()
            elif strategy == 'mode':
                fill_value = processed_df[col].mode()[0]
            else:
                fill_value = 0
            processed_df[col].fillna(fill_value, inplace=True)
    return processed_df

def clean_dataset(dataframe, numeric_columns):
    df_no_missing = handle_missing_values(dataframe, numeric_columns, 'median')
    df_no_outliers = remove_outliers_iqr(df_no_missing, numeric_columns)
    df_normalized = normalize_data(df_no_outliers, numeric_columns, 'zscore')
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature3': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    }
    df = pd.DataFrame(sample_data)
    numeric_cols = ['feature1', 'feature2', 'feature3']
    cleaned_df = clean_dataset(df, numeric_cols)
    print("Original dataset shape:", df.shape)
    print("Cleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned dataset summary:")
    print(cleaned_df.describe())
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
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Parameters:
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
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', None, 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid}, Message: {message}")import pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
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
                if dtype == 'numeric':
                    df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
                elif dtype == 'datetime':
                    df_converted[column] = pd.to_datetime(df_converted[column], errors='coerce')
    
    return df_converted

def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'drop',
                         fill_value: Union[int, float, str] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: Value to use when filling missing values
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill' and fill_value is not None:
        return df.fillna(fill_value)
    else:
        return df

def clean_dataframe(df: pd.DataFrame,
                   deduplicate: bool = True,
                   type_conversions: dict = None,
                   missing_strategy: str = 'drop',
                   fill_value: Union[int, float, str] = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        type_conversions: Dictionary for column type conversions
        missing_strategy: Strategy for handling missing values
        fill_value: Value to fill missing values with
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, missing_strategy, fill_value)
    
    return cleaned_df

def validate_dataframe(df: pd.DataFrame,
                      required_columns: List[str] = None,
                      min_rows: int = 1) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        min_rows: Minimum number of rows required
    
    Returns:
        Boolean indicating if DataFrame is valid
    """
    if df.empty:
        return False
    
    if len(df) < min_rows:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None],
        'age': ['25', '30', '30', 'thirty-five', '40'],
        'score': [85.5, 92.0, 92.0, None, 78.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    cleaned = clean_dataframe(
        df,
        deduplicate=True,
        type_conversions={'age': 'numeric', 'score': 'float'},
        missing_strategy='fill',
        fill_value=0
    )
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nCleaned DataFrame info:")
    print(cleaned.info())
    
    is_valid = validate_dataframe(cleaned, required_columns=['id', 'name', 'age'])
    print(f"\nDataFrame is valid: {is_valid}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column using min-max scaling
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
    Standardize column using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
    elif strategy == 'mode':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
    elif strategy == 'drop':
        data = data.dropna(subset=numeric_cols)
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    return data

def create_clean_dataframe(data, outlier_columns=None, normalize_columns=None, 
                          standardize_columns=None, missing_strategy='mean'):
    """
    Apply multiple cleaning operations to a DataFrame
    """
    cleaned_data = data.copy()
    
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    if normalize_columns:
        for col in normalize_columns:
            if col in cleaned_data.columns:
                cleaned_data[col + '_normalized'] = normalize_minmax(cleaned_data, col)
    
    if standardize_columns:
        for col in standardize_columns:
            if col in cleaned_data.columns:
                cleaned_data[col + '_standardized'] = standardize_zscore(cleaned_data, col)
    
    return cleaned_data
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_missing(self, threshold=0.3):
        missing_percent = self.df.isnull().sum() / len(self.df)
        columns_to_drop = missing_percent[missing_percent > threshold].index
        self.df = self.df.drop(columns=columns_to_drop)
        return self
    
    def fill_numeric_missing(self, method='median'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if method == 'median':
                    fill_value = self.df[col].median()
                elif method == 'mean':
                    fill_value = self.df[col].mean()
                elif method == 'mode':
                    fill_value = self.df[col].mode()[0]
                else:
                    fill_value = 0
                
                self.df[col] = self.df[col].fillna(fill_value)
        
        return self
    
    def detect_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_mask = pd.Series([False] * len(self.df))
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            col_outliers = z_scores > threshold
            outliers_mask = outliers_mask | col_outliers.reindex(self.df.index, fill_value=False)
        
        return outliers_mask
    
    def remove_outliers(self, threshold=3):
        outliers_mask = self.detect_outliers_zscore(threshold)
        self.df = self.df[~outliers_mask]
        return self
    
    def normalize_data(self, method='minmax'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val != 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_cleaning_report(self):
        rows_removed = self.original_shape[0] - len(self.df)
        cols_removed = self.original_shape[1] - self.df.shape[1]
        
        report = {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'missing_values': self.df.isnull().sum().sum()
        }
        
        return report

def clean_dataset(df, outlier_threshold=3, normalize=True):
    cleaner = DataCleaner(df)
    
    cleaner.remove_missing(threshold=0.3)
    cleaner.fill_numeric_missing(method='median')
    cleaner.remove_outliers(threshold=outlier_threshold)
    
    if normalize:
        cleaner.normalize_data(method='minmax')
    
    return cleaner.get_cleaned_data(), cleaner.get_cleaning_report()