
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: A list of elements (must be hashable)
    
    Returns:
        A new list with duplicates removed
    """
    seen = set()
    result = []
    
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_data_with_key(data_list, key_func=None):
    """
    Remove duplicates based on a key function.
    
    Args:
        data_list: A list of elements
        key_func: Function to extract comparison key (default: identity)
    
    Returns:
        A new list with duplicates removed based on key
    """
    if key_func is None:
        return remove_duplicates(data_list)
    
    seen = set()
    result = []
    
    for item in data_list:
        key = key_func(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    
    return result

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5, 1]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    # Example with custom key
    data_dicts = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 1, "name": "Alice"},
        {"id": 3, "name": "Charlie"}
    ]
    
    cleaned_dicts = clean_data_with_key(data_dicts, key_func=lambda x: x["id"])
    print(f"\nOriginal dicts: {data_dicts}")
    print(f"Cleaned dicts: {cleaned_dicts}")
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_strings(string_list):
    """
    Clean a list of numeric strings by converting to integers,
    removing duplicates, and returning sorted unique integers.
    """
    unique_numbers = set()
    for s in string_list:
        try:
            num = int(s.strip())
            unique_numbers.add(num)
        except ValueError:
            continue
    return sorted(unique_numbers)

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    numeric_strings = ["10", "5", "3", "5", "20", "3", "abc"]
    cleaned_nums = clean_numeric_strings(numeric_strings)
    print(f"Numeric strings cleaned: {cleaned_nums}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    return filtered_df

def zscore_normalize(dataframe, column):
    """
    Normalize a column using z-score normalization.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized column values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    
    if std_val == 0:
        return dataframe[column]
    
    normalized = (dataframe[column] - mean_val) / std_val
    return normalized

def minmax_normalize(dataframe, column, feature_range=(0, 1)):
    """
    Normalize a column using min-max normalization.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    feature_range (tuple): Desired range of transformed data
    
    Returns:
    pd.Series: Normalized column values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if min_val == max_val:
        return dataframe[column]
    
    a, b = feature_range
    normalized = a + ((dataframe[column] - min_val) * (b - a)) / (max_val - min_val)
    return normalized

def detect_skewness(dataframe, column):
    """
    Detect skewness of a column and suggest transformation.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Skewness statistics and transformation suggestion
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    skew_value = dataframe[column].skew()
    
    result = {
        'skewness': skew_value,
        'abs_skewness': abs(skew_value),
        'suggestion': 'No transformation needed'
    }
    
    if abs(skew_value) > 1:
        result['suggestion'] = 'Strongly skewed - consider log or box-cox transformation'
    elif abs(skew_value) > 0.5:
        result['suggestion'] = 'Moderately skewed - consider square root transformation'
    
    return result

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric columns to process
    outlier_threshold (float): IQR threshold for outlier removal
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_threshold)
            cleaned_df[f'{column}_normalized'] = zscore_normalize(cleaned_df, column)
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    dataframe (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    if not isinstance(dataframe, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append('Input is not a pandas DataFrame')
        return validation_results
    
    if dataframe.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('DataFrame is empty')
        return validation_results
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    validation_results['stats']['shape'] = dataframe.shape
    validation_results['stats']['missing_values'] = dataframe.isnull().sum().to_dict()
    validation_results['stats']['dtypes'] = dataframe.dtypes.to_dict()
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        validation_results['stats']['numeric_summary'] = dataframe[numeric_cols].describe().to_dict()
    
    return validation_results
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str, optional): Path for cleaned output CSV
    missing_strategy (str): Strategy for handling missing values
                           ('mean', 'median', 'drop', 'zero')
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    # Validate input file exists
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read CSV file
    df = pd.read_csv(input_path)
    
    # Remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif missing_strategy == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    elif missing_strategy == 'zero':
        df = df.fillna(0)
    elif missing_strategy == 'drop':
        df = df.dropna()
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    # Save cleaned data if output path provided
    if output_path:
        output_file = Path(output_path)
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    # Print cleaning summary
    print(f"Data cleaning completed:")
    print(f"  - Initial rows: {initial_rows}")
    print(f"  - Duplicates removed: {duplicates_removed}")
    print(f"  - Final rows: {len(df)}")
    print(f"  - Missing strategy: {missing_strategy}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate
    required_columns (list, optional): List of required column names
    
    Returns:
    dict: Validation results
    """
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_cols}')
    
    # Check for all-null columns
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        validation_results['warnings'].append(f'Columns with all null values: {null_columns}')
    
    # Check data types
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            if unique_count == len(df):
                validation_results['warnings'].append(f'Column "{col}" has all unique values (possible ID column)')
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, np.nan, 15.2, np.nan, 20.1, 20.1],
        'category': ['A', 'B', 'A', 'C', 'B', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    # Clean the sample data
    cleaned_df = clean_csv_data('sample_data.csv', 'cleaned_sample.csv', 'mean')
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
    print(f"\nValidation results: {validation}")