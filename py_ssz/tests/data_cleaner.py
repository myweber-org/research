
import pandas as pd
import numpy as np

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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    })
    
    numeric_cols = ['feature_a', 'feature_b']
    cleaned = clean_dataset(sample_data, numeric_cols)
    save_cleaned_data(cleaned, 'cleaned_data.csv')import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Data validation passed"

def process_data(file_path, output_path=None):
    """
    Load, clean, and optionally save a dataset.
    
    Args:
        file_path (str): Path to input data file.
        output_path (str, optional): Path to save cleaned data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    
    is_valid, message = validate_data(df)
    if not is_valid:
        raise ValueError(f"Data validation failed: {message}")
    
    cleaned_df = clean_dataset(df)
    
    if output_path:
        cleaned_df.to_csv(output_path, index=False)
    
    return cleaned_df
import numpy as np
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
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling to range [0, 1].
    
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
    Standardize data using z-score normalization.
    
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

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names (default: all numeric columns)
        outlier_multiplier: IQR multiplier for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in data.columns:
            # Remove outliers
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_multiplier)
            
            # Normalize the column
            cleaned_data[f"{column}_normalized"] = normalize_minmax(cleaned_data, column)
            cleaned_data[f"{column}_standardized"] = standardize_zscore(cleaned_data, column)
    
    return cleaned_data

def get_summary_statistics(data, column):
    """
    Get summary statistics for a column.
    
    Args:
        data: pandas DataFrame
        column: column name
    
    Returns:
        Dictionary with summary statistics
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'q1': data[column].quantile(0.25),
        'q3': data[column].quantile(0.75),
        'count': data[column].count(),
        'missing': data[column].isnull().sum()
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("\nFeature1 statistics:")
    print(get_summary_statistics(sample_data, 'feature1'))
    
    cleaned = clean_dataset(sample_data, ['feature1', 'feature2'])
    print("\nCleaned data shape:", cleaned.shape)
    print("\nAvailable columns:", cleaned.columns.tolist())
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return dataframe[(dataframe[column] >= lower_bound) & 
                     (dataframe[column] <= upper_bound)]

def zscore_normalize(dataframe, column):
    """
    Normalize column using z-score normalization
    """
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    
    if std_val > 0:
        dataframe[column + '_normalized'] = (dataframe[column] - mean_val) / std_val
    else:
        dataframe[column + '_normalized'] = 0
    
    return dataframe

def minmax_normalize(dataframe, column, feature_range=(0, 1)):
    """
    Normalize column using min-max normalization
    """
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if max_val > min_val:
        dataframe[column + '_scaled'] = (
            (dataframe[column] - min_val) / (max_val - min_val) * 
            (feature_range[1] - feature_range[0]) + feature_range[0]
        )
    else:
        dataframe[column + '_scaled'] = feature_range[0]
    
    return dataframe

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = dataframe.columns
    
    df_copy = dataframe.copy()
    
    for col in columns:
        if df_copy[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_copy[col].mean()
            elif strategy == 'median':
                fill_value = df_copy[col].median()
            elif strategy == 'mode':
                fill_value = df_copy[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', 'constant'")
            
            df_copy[col].fillna(fill_value, inplace=True)
    
    return df_copy

def clean_dataset(dataframe, numeric_columns=None, outlier_multiplier=1.5, 
                  normalize_method='zscore', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = dataframe.copy()
    
    # Handle missing values
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy, columns=numeric_columns)
    
    # Remove outliers for each numeric column
    for col in numeric_columns:
        df_clean = remove_outliers_iqr(df_clean, col, multiplier=outlier_multiplier)
    
    # Normalize numeric columns
    for col in numeric_columns:
        if normalize_method == 'zscore':
            df_clean = zscore_normalize(df_clean, col)
        elif normalize_method == 'minmax':
            df_clean = minmax_normalize(df_clean, col)
    
    return df_clean

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content
    """
    if len(dataframe) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True
import re
import pandas as pd
from typing import List, Optional, Union

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def standardize_strings(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Standardize strings in specified columns: strip whitespace and convert to lowercase.
    """
    df_copy = df.copy()
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.strip().str.lower()
    return df_copy

def validate_email_column(df: pd.DataFrame, email_column: str) -> pd.Series:
    """
    Validate email addresses in a specified column using a regex pattern.
    Returns a boolean Series indicating valid emails.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    return df[email_column].astype(str).str.match(pattern)

def fill_missing_with_mean(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """
    Fill missing values in numeric columns with the column mean.
    """
    df_copy = df.copy()
    for col in numeric_columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            mean_val = df_copy[col].mean()
            df_copy[col].fillna(mean_val, inplace=True)
    return df_copy

def filter_by_threshold(df: pd.DataFrame, column: str, threshold: float, keep_above: bool = True) -> pd.DataFrame:
    """
    Filter rows based on a threshold value in a numeric column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if keep_above:
        return df[df[column] >= threshold].copy()
    else:
        return df[df[column] <= threshold].copy()import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, overwrites input file
        subset (list, optional): Columns to consider for identifying duplicates
        keep (str): Which duplicates to keep - 'first', 'last', or False to drop all
        
    Returns:
        int: Number of duplicate rows removed
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        final_rows = len(df_clean)
        
        duplicates_removed = initial_rows - final_rows
        
        if output_file is None:
            output_file = input_file
            
        df_clean.to_csv(output_file, index=False)
        
        print(f"Removed {duplicates_removed} duplicate rows")
        print(f"Original rows: {initial_rows}")
        print(f"Cleaned rows: {final_rows}")
        print(f"Output saved to: {output_file}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return -1
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty")
        return -1
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return -1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = remove_duplicates(input_file, output_file)
    
    if result >= 0:
        sys.exit(0)
    else:
        sys.exit(1)