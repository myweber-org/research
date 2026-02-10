
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport numpy as np
import pandas as pd
from scipy import stats

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

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

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
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[col] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def get_data_summary(df):
    """
    Generate statistical summary of the dataset
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    for col in df.select_dtypes(include=[np.number]).columns:
        summary[f'{col}_stats'] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'q1': df[col].quantile(0.25),
            'q3': df[col].quantile(0.75)
        }
    
    return summary

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content
    """
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(df) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows, got {len(df)}")
    
    if df.isnull().all().any():
        raise ValueError("Dataset contains columns with all null values")
    
    return Trueimport pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_clean (list, optional): List of column names to apply string normalization.
            If None, all object dtype columns are cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Determine columns to normalize
    if columns_to_clean is None:
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    # Normalize string columns
    for col in columns_to_clean:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
    
    return cleaned_df

def _normalize_string(s):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters from the edges.
    
    Args:
        s (str): Input string.
    
    Returns:
        str: Normalized string.
    """
    if not isinstance(s, str):
        return s
    
    # Convert to lowercase
    s = s.lower()
    
    # Remove leading/trailing whitespace
    s = s.strip()
    
    # Replace multiple spaces with single space
    s = re.sub(r'\s+', ' ', s)
    
    return s

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    validated_df = df.copy()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validated_df['email_valid'] = validated_df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    return validated_df

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'name': ['John Doe', 'Jane Smith', 'John Doe', '  Bob Johnson  '],
#         'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.co.uk'],
#         'age': [25, 30, 25, 35]
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataframe(df)
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     validated = validate_email_column(cleaned, 'email')
#     print("\nDataFrame with email validation:")
#     print(validated)
import pandas as pd

def clean_dataframe(df, fill_strategy='drop', column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    fill_strategy (str): Strategy for handling null values. 
                         Options: 'drop' to drop rows, 'fill' to fill with column mean (numeric) or mode (object).
    column_case (str): Target case for column names. Options: 'lower', 'upper', 'title'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Standardize column names
    if column_case == 'lower':
        df_clean.columns = df_clean.columns.str.lower()
    elif column_case == 'upper':
        df_clean.columns = df_clean.columns.str.upper()
    elif column_case == 'title':
        df_clean.columns = df_clean.columns.str.title()
    
    # Handle missing values
    if fill_strategy == 'drop':
        df_clean = df_clean.dropna()
    elif fill_strategy == 'fill':
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else '', inplace=True)
    
    # Reset index after cleaning
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (bool, str) indicating validation result and message.
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame validation passed"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {'Name': ['Alice', 'Bob', None], 'Age': [25, None, 30], 'Score': [85.5, 92.0, 88.5]}
#     df = pd.DataFrame(sample_data)
#     cleaned_df = clean_dataframe(df, fill_strategy='fill', column_case='title')
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)
#     validation_result, message = validate_dataframe(cleaned_df, required_columns=['Name', 'Age', 'Score'])
#     print(f"\nValidation: {message}")
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns indices of outliers.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers.index.tolist()

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    Returns cleaned dataframe.
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column using Min-Max scaling.
    Returns normalized series.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_standard(data, column):
    """
    Normalize column using Standardization (Z-score normalization).
    Returns normalized series.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    if columns is None:
        columns = data.columns
    
    data_clean = data.copy()
    
    for col in columns:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'drop':
                data_clean = data_clean.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            data_clean[col] = data_clean[col].fillna(fill_value)
    
    return data_clean

def clean_dataset(data, outlier_method='iqr', normalize_method=None, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_data = data.copy()
    
    numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            outliers = detect_outliers_iqr(cleaned_data, col)
            if outliers:
                cleaned_data = cleaned_data.drop(index=outliers)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    if normalize_method:
        for col in numeric_cols:
            if normalize_method == 'minmax':
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
            elif normalize_method == 'standard':
                cleaned_data[col] = normalize_standard(cleaned_data, col)
    
    return cleaned_data.reset_index(drop=True)
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if cleaned_df.isnull().sum().sum() > 0:
        print(f"Found {cleaned_df.isnull().sum().sum()} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values")
        elif fill_missing in ['mean', 'median']:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if fill_missing == 'mean':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                else:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            print(f"Filled missing numeric values with {fill_missing}")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0)
            print("Filled missing values with mode")
    
    print(f"Cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append("Input is not a pandas DataFrame")
        return validation_results
    
    if df.empty:
        validation_results['warnings'].append("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        validation_results['warnings'].append("No numeric columns found")
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataframe(df, required_columns=['A', 'B'])
    print("Validation results:", validation)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
import pandas as pd
import numpy as np
from typing import Optional, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
    
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset)
        return self
    
    def fill_missing_values(self, strategy: str = 'mean', fill_value: Optional[float] = None) -> 'DataCleaner':
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mode().iloc[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_value)
        
        return self
    
    def remove_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> 'DataCleaner':
        if method == 'iqr':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        return self
    
    def normalize_data(self, method: str = 'minmax') -> 'DataCleaner':
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
    def get_cleaning_stats(self) -> dict:
        final_shape = self.df.shape
        rows_removed = self.original_shape[0] - final_shape[0]
        cols_removed = self.original_shape[1] - final_shape[1]
        
        return {
            'original_rows': self.original_shape[0],
            'original_cols': self.original_shape[1],
            'final_rows': final_shape[0],
            'final_cols': final_shape[1],
            'rows_removed': rows_removed,
            'cols_removed': cols_removed,
            'missing_values': self.df.isnull().sum().sum()
        }

def load_and_clean_csv(filepath: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(filepath, **kwargs)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates()
    cleaner.fill_missing_values(strategy='mean')
    cleaner.remove_outliers()
    
    return cleaner.get_cleaned_data()
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns, method='minmax'):
    df_norm = df.copy()
    for col in columns:
        if method == 'minmax':
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            df_norm[col] = (df_norm[col] - mean_val) / std_val
    return df_norm

def handle_missing_values(df, columns, strategy='mean'):
    df_filled = df.copy()
    for col in columns:
        if strategy == 'mean':
            fill_value = df_filled[col].mean()
        elif strategy == 'median':
            fill_value = df_filled[col].median()
        elif strategy == 'mode':
            fill_value = df_filled[col].mode()[0]
        df_filled[col].fillna(fill_value, inplace=True)
    return df_filled

def clean_dataset(df, numeric_columns):
    df_processed = df.copy()
    df_processed = handle_missing_values(df_processed, numeric_columns)
    df_processed = remove_outliers_iqr(df_processed, numeric_columns)
    df_processed = normalize_data(df_processed, numeric_columns)
    return df_processedimport numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling to range [0, 1].
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    result = data.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if strategy == 'drop':
            result = result.dropna(subset=[col])
        elif strategy == 'mean':
            result[col] = result[col].fillna(result[col].mean())
        elif strategy == 'median':
            result[col] = result[col].fillna(result[col].median())
        elif strategy == 'mode':
            result[col] = result[col].fillna(result[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return result

def create_data_summary(data):
    """
    Create a summary statistics DataFrame.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        DataFrame with summary statistics
    """
    summary = pd.DataFrame({
        'dtype': data.dtypes,
        'non_null_count': data.count(),
        'null_count': data.isnull().sum(),
        'null_percentage': (data.isnull().sum() / len(data)) * 100,
        'unique_count': data.nunique()
    })
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = data[numeric_cols].describe().T
        summary = summary.join(numeric_stats, how='left')
    
    return summary
import pandas as pd
import numpy as np

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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
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
        'count': len(df[column]),
        'q1': df[column].quantile(0.25),
        'q3': df[column].quantile(0.75)
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not process column '{col}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[0, 'A'] = 1000
    df.loc[1, 'B'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary for column 'A':")
    print(calculate_summary_stats(df, 'A'))
    
    cleaned_df = clean_numeric_data(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned summary for column 'A':")
    print(calculate_summary_stats(cleaned_df, 'A'))