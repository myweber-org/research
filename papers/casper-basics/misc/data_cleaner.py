import csv
import re
from typing import List, Dict, Any

def remove_duplicates(data: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = set()
    unique_data = []
    for row in data:
        if row[key] not in seen:
            seen.add(row[key])
            unique_data.append(row)
    return unique_data

def normalize_string(value: str) -> str:
    if not isinstance(value, str):
        return value
    value = value.strip().lower()
    value = re.sub(r'\s+', ' ', value)
    return value

def clean_numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r'[^\d.-]', '', value)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0

def read_csv_file(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

def write_csv_file(data: List[Dict[str, Any]], filepath: str, fieldnames: List[str]) -> None:
    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def clean_csv_data(input_file: str, output_file: str, unique_key: str) -> None:
    data = read_csv_file(input_file)
    
    for row in data:
        for key, value in row.items():
            if isinstance(value, str):
                row[key] = normalize_string(value)
            elif key.endswith('_amount') or key.endswith('_price'):
                row[key] = clean_numeric(value)
    
    cleaned_data = remove_duplicates(data, unique_key)
    
    if cleaned_data:
        fieldnames = list(cleaned_data[0].keys())
        write_csv_file(cleaned_data, output_file, fieldnames)
        print(f"Cleaned data saved to {output_file}")
    else:
        print("No data to save")

if __name__ == "__main__":
    clean_csv_data("raw_data.csv", "cleaned_data.csv", "id")import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str or dict): Method to fill missing values.
    
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
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        else:
            cleaned_df = cleaned_df.fillna(fill_missing)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
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

def process_data(file_path, output_path=None, **clean_kwargs):
    """
    Load, clean, and optionally save data.
    
    Parameters:
    file_path (str): Path to input data file.
    output_path (str): Path to save cleaned data.
    **clean_kwargs: Additional arguments for clean_dataframe.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        if not validate_dataframe(df):
            raise ValueError("Data validation failed")
        
        cleaned_df = clean_dataframe(df, **clean_kwargs)
        
        if output_path:
            if output_path.endswith('.csv'):
                cleaned_df.to_csv(output_path, index=False)
            elif output_path.endswith('.xlsx'):
                cleaned_df.to_excel(output_path, index=False)
        
        return cleaned_df
    
    except Exception as e:
        print(f"Error processing data: {e}")
        return None
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    cleaned_df = df.copy()

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {original_shape[0] - cleaned_df.shape[0]} duplicate rows.")

    if cleaned_df.isnull().sum().any():
        print("Handling missing values...")
        for column in cleaned_df.columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    fill_value = cleaned_df[column].mean()
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    fill_value = cleaned_df[column].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                else:
                    fill_value = 0 if pd.api.types.is_numeric_dtype(cleaned_df[column]) else 'Unknown'
                cleaned_df[column].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{column}' with {fill_value}.")

    print(f"Data cleaning complete. Original shape: {original_shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate the DataFrame for required columns and data types.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    for column in df.columns:
        if df[column].dtype == 'object':
            unique_count = df[column].nunique()
            if unique_count / len(df) > 0.5:
                print(f"Warning: Column '{column}' has high cardinality ({unique_count} unique values).")

    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, np.nan, 15.0, 20.0, np.nan, 30.0],
        'category': ['A', 'B', 'B', 'A', 'C', 'A']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a pandas Series using the IQR method.
    
    Parameters:
    data (pd.Series): Input data series
    column (str): Column name to process
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    series = data[column]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(series >= lower_bound) & (series <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized series
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    series = data[column]
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    
    normalized = (series - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to standardize
    
    Returns:
    pd.Series: Standardized series
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    series = data[column]
    mean_val = series.mean()
    std_val = series.std()
    
    if std_val == 0:
        return pd.Series([0] * len(series), index=series.index)
    
    standardized = (series - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to process
    outlier_factor (float): Factor for IQR outlier detection
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in df.columns:
            # Remove outliers
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            
            # Normalize the column
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage demonstration
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Add some outliers
    sample_data.loc[0, 'feature_a'] = 500
    sample_data.loc[1, 'feature_b'] = 1000
    
    print("Original data shape:", sample_data.shape)
    print("Original data summary:")
    print(sample_data[['feature_a', 'feature_b']].describe())
    
    # Clean the data
    cleaned_data = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("Cleaned data summary:")
    print(cleaned_data[['feature_a', 'feature_b']].describe())
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned_data, ['feature_a', 'feature_b'])
    print(f"\nData validation: {message}")import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    # Handle missing values
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print(f"Dropped rows with missing values")
        elif fill_missing == 'mean':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
            print(f"Filled numeric columns with mean values")
        elif fill_missing == 'median':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
            print(f"Filled numeric columns with median values")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
            print(f"Filled categorical columns with mode values")
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    print(f"Cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    sample_data = {
        'id': [1, 2, 3, 2, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', 'Bob', 'Eve', None],
        'age': [25, 30, None, 30, 35, 40],
        'score': [85.5, 92.0, 78.5, 92.0, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid = validate_dataframe(cleaned, required_columns=['id', 'name', 'age', 'score'], min_rows=1)
    print(f"\nData validation: {'PASSED' if is_valid else 'FAILED'}")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_value: Value to use for filling missing data
    
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
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        missing_after = cleaned_df.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate a summary of the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': df.describe().to_dict() if df.select_dtypes(include=['number']).shape[1] > 0 else {}
    }
    return summary

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 3, None],
        'B': [4, 5, 5, None, 7],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nData Summary:")
    print(get_data_summary(df))
    
    cleaned = clean_dataset(df, fill_value=0)
    print("\nCleaned DataFrame:")
    print(cleaned)
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    })
    sample_data.loc[::100, 'A'] = 500
    numeric_cols = ['A', 'B', 'C']
    result = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {result.shape}")
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    """
    if subset:
        return df.drop_duplicates(subset=subset, keep='first')
    return df.drop_duplicates(keep='first')

def convert_column_types(df, column_types):
    """
    Convert columns to specified data types.
    """
    for column, dtype in column_types.items():
        if column in df.columns:
            df[column] = df[column].astype(dtype)
    return df

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            return df.fillna(fill_value)
        else:
            return df.fillna(df.mean())
    return df

def normalize_column(df, column):
    """
    Normalize a column to range [0, 1].
    """
    if column in df.columns:
        col_min = df[column].min()
        col_max = df[column].max()
        if col_max != col_min:
            df[column] = (df[column] - col_min) / (col_max - col_min)
    return df

def clean_dataframe(df, config):
    """
    Apply multiple cleaning operations based on configuration.
    """
    if 'remove_duplicates' in config:
        df = remove_duplicates(df, config.get('remove_duplicates'))
    
    if 'column_types' in config:
        df = convert_column_types(df, config['column_types'])
    
    if 'missing_values' in config:
        missing_config = config['missing_values']
        df = handle_missing_values(
            df, 
            strategy=missing_config.get('strategy', 'drop'),
            fill_value=missing_config.get('fill_value')
        )
    
    if 'normalize_columns' in config:
        for column in config['normalize_columns']:
            df = normalize_column(df, column)
    
    return df

def validate_dataframe(df, rules):
    """
    Validate DataFrame against a set of rules.
    """
    violations = []
    
    for rule in rules:
        if rule['type'] == 'not_null':
            null_count = df[rule['column']].isnull().sum()
            if null_count > 0:
                violations.append(f"Column {rule['column']} has {null_count} null values")
        
        elif rule['type'] == 'unique':
            duplicate_count = df[rule['column']].duplicated().sum()
            if duplicate_count > 0:
                violations.append(f"Column {rule['column']} has {duplicate_count} duplicate values")
        
        elif rule['type'] == 'range':
            min_val = rule.get('min')
            max_val = rule.get('max')
            if min_val is not None:
                below_min = (df[rule['column']] < min_val).sum()
                if below_min > 0:
                    violations.append(f"Column {rule['column']} has {below_min} values below {min_val}")
            if max_val is not None:
                above_max = (df[rule['column']] > max_val).sum()
                if above_max > 0:
                    violations.append(f"Column {rule['column']} has {above_max} values above {max_val}")
    
    return violations
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): List of columns to check for duplicates.
            If None, checks all columns. Defaults to None.
        fill_missing (str, optional): Method to fill missing values.
            Options: 'mean', 'median', 'mode', or 'drop'. Defaults to 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    else:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    # For non-numeric columns, fill with the most frequent value
    non_numeric_cols = cleaned_df.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
        min_rows (int, optional): Minimum number of rows required.
    
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

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4, 5],
#         'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
#         'age': [25, 30, 30, None, 35, 40],
#         'score': [85.5, 90.0, 90.0, 75.5, None, 95.0]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame (mean imputation):")
#     cleaned = clean_dataset(df, columns_to_check=['id'], fill_missing='mean')
#     print(cleaned)
#     
#     # Validate the cleaned data
#     is_valid, message = validate_data(cleaned, required_columns=['id', 'name', 'age'])
#     print(f"\nValidation: {is_valid} - {message}")import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame using specified strategy.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values. 
                       Options: 'mean', 'median', 'mode', 'drop', 'fill_zero'
        columns (list): List of columns to apply cleaning. If None, applies to all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if strategy == 'mean':
            fill_value = df_clean[col].mean()
        elif strategy == 'median':
            fill_value = df_clean[col].median()
        elif strategy == 'mode':
            fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
        elif strategy == 'fill_zero':
            fill_value = 0
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to standardize
    
    Returns:
        pd.DataFrame: Standardized DataFrame
    """
    from sklearn.preprocessing import StandardScaler
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    scaler = StandardScaler()
    
    df_clean[columns] = scaler.fit_transform(df_clean[columns])
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): Columns that must be present
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, np.nan, 30, 40, 50, 60],
        'C': ['a', 'b', 'c', 'd', 'e', 'f']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (mean imputation):")
    cleaned_df = clean_missing_data(df, strategy='mean', columns=['A', 'B'])
    print(cleaned_df)
    
    print("\nOutlier removal:")
    outlier_free = remove_outliers_iqr(cleaned_df, columns=['A', 'B'])
    print(outlier_free)
    
    print("\nStandardized data:")
    standardized = standardize_columns(outlier_free, columns=['A', 'B'])
    print(standardized)
    
    print("\nValidation result:", validate_dataframe(standardized, min_rows=1))
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
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
    
    return filtered_df

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If columns is None, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not clean column '{column}': {e}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        tuple: (is_valid, message)
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

def main():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    
    is_valid, message = validate_dataframe(df, ['id', 'value'])
    print(f"Validation: {message}")
    
    cleaned_df = clean_numeric_data(df, columns=['value'])
    print("Cleaned DataFrame shape:", cleaned_df.shape)
    
    print(f"Removed {len(df) - len(cleaned_df)} outliers")

if __name__ == "__main__":
    main()