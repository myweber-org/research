
import numpy as np
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

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f'{column}_standardized'] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, check_missing=True, check_duplicates=True):
    """
    Validate dataset for common data quality issues
    """
    validation_report = {}
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_report['missing_columns'] = missing_columns
    
    if check_missing:
        missing_values = data.isnull().sum()
        missing_percentage = (missing_values / len(data)) * 100
        validation_report['missing_values'] = missing_percentage[missing_percentage > 0].to_dict()
    
    if check_duplicates:
        duplicate_count = data.duplicated().sum()
        validation_report['duplicate_rows'] = duplicate_count
    
    return validation_report
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to remove duplicate rows.
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

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
        method (str): 'iqr' for interquartile range or 'zscore' for standard deviations.
        threshold (float): Threshold value for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data = df[column].copy()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        z_scores = (data - mean) / std
        mask = abs(z_scores) <= threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]

def normalize_column(df, column, method='minmax'):
    """
    Normalize values in a specific column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to normalize.
        method (str): 'minmax' for min-max scaling or 'standard' for standardization.
    
    Returns:
        pd.DataFrame: DataFrame with normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    normalized_df = df.copy()
    data = normalized_df[column]
    
    if method == 'minmax':
        min_val = data.min()
        max_val = data.max()
        if max_val != min_val:
            normalized_df[column] = (data - min_val) / (max_val - min_val)
    
    elif method == 'standard':
        mean = data.mean()
        std = data.std()
        if std != 0:
            normalized_df[column] = (data - mean) / std
    
    else:
        raise ValueError("Method must be 'minmax' or 'standard'")
    
    return normalized_df

def validate_dataframe(df, required_columns=None, allow_nan=False):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
        allow_nan (bool): Whether NaN values are allowed.
    
    Returns:
        tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if not allow_nan and df.isnull().any().any():
        return False, "DataFrame contains NaN values"
    
    return True, "DataFrame is valid"import pandas as pd
import numpy as np

def clean_csv_data(filepath, drop_na=True, fill_strategy='mean'):
    """
    Load and clean a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        drop_na (bool): If True, drop rows with any NaN values.
                        If False, fill NaN values based on fill_strategy.
        fill_strategy (str): Strategy to fill NaN values if drop_na is False.
                             Options: 'mean', 'median', 'zero', 'ffill'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()
    
    original_rows = len(df)
    original_cols = len(df.columns)
    
    if drop_na:
        df_cleaned = df.dropna()
        rows_removed = original_rows - len(df_cleaned)
        print(f"Dropped {rows_removed} rows with NaN values.")
    else:
        df_cleaned = df.copy()
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if fill_strategy == 'mean':
                fill_value = df_cleaned[col].mean()
            elif fill_strategy == 'median':
                fill_value = df_cleaned[col].median()
            elif fill_strategy == 'zero':
                fill_value = 0
            elif fill_strategy == 'ffill':
                df_cleaned[col] = df_cleaned[col].ffill()
                continue
            else:
                fill_value = df_cleaned[col].mean()
            
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
        
        print(f"Filled NaN values using '{fill_strategy}' strategy.")
    
    string_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in string_cols:
        df_cleaned[col] = df_cleaned[col].str.strip()
    
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    print(f"Original data: {original_rows} rows, {original_cols} columns")
    print(f"Cleaned data: {len(df_cleaned)} rows, {len(df_cleaned.columns)} columns")
    
    return df_cleaned

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list): Columns to consider for identifying duplicates.
        keep (str): Which duplicates to keep. Options: 'first', 'last', False.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    original_rows = len(df)
    df_deduped = df.drop_duplicates(subset=subset, keep=keep)
    rows_removed = original_rows - len(df_deduped)
    
    print(f"Removed {rows_removed} duplicate rows.")
    
    return df_deduped

def validate_data_types(df, expected_types):
    """
    Validate that columns have expected data types.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        expected_types (dict): Dictionary mapping column names to expected types.
    
    Returns:
        dict: Dictionary with validation results.
    """
    validation_results = {}
    
    for column, expected_type in expected_types.items():
        if column not in df.columns:
            validation_results[column] = {
                'valid': False,
                'message': f"Column '{column}' not found in DataFrame."
            }
            continue
        
        actual_type = str(df[column].dtype)
        
        if expected_type == 'numeric':
            is_valid = np.issubdtype(df[column].dtype, np.number)
            type_name = 'numeric'
        elif expected_type == 'string':
            is_valid = df[column].dtype == 'object'
            type_name = 'string'
        else:
            is_valid = actual_type == expected_type
            type_name = expected_type
        
        validation_results[column] = {
            'valid': is_valid,
            'expected': type_name,
            'actual': actual_type
        }
    
    return validation_results
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
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
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Error processing column {column}: {e}")
                continue
    
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
    
    print("Original dataset shape:", df.shape)
    print("Original statistics for column A:", calculate_basic_stats(df, 'A'))
    
    cleaned_df = clean_dataset(df, ['A', 'B'])
    
    print("Cleaned dataset shape:", cleaned_df.shape)
    print("Cleaned statistics for column A:", calculate_basic_stats(cleaned_df, 'A'))
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result