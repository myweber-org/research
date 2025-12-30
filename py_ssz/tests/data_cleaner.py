
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data (np.ndarray): Input data array
        column (int): Column index to process
    
    Returns:
        np.ndarray: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    column_data = data[:, column]
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Args:
        data (np.ndarray): Input data array
        column (int): Column index to analyze
    
    Returns:
        dict: Dictionary containing statistical measures
    """
    cleaned_data = remove_outliers_iqr(data, column)
    column_data = cleaned_data[:, column]
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data),
        'count': len(column_data)
    }
    
    return stats

def validate_data_shape(data, expected_columns):
    """
    Validate that data has the expected number of columns.
    
    Args:
        data (np.ndarray): Input data array
        expected_columns (int): Expected number of columns
    
    Returns:
        bool: True if shape is valid, False otherwise
    """
    if len(data.shape) != 2:
        return False
    
    return data.shape[1] == expected_columns

def example_usage():
    """
    Demonstrate usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    sample_data = np.random.randn(100, 3)
    sample_data[:, 1] = sample_data[:, 1] * 10 + 5
    
    print("Original data shape:", sample_data.shape)
    
    cleaned_data = remove_outliers_iqr(sample_data, 1)
    print("Cleaned data shape:", cleaned_data.shape)
    
    stats = calculate_statistics(sample_data, 1)
    print("Statistics for column 1:", stats)
    
    is_valid = validate_data_shape(sample_data, 3)
    print("Data shape validation:", is_valid)

if __name__ == "__main__":
    example_usage()
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        Boolean indicating if validation passed
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate summary statistics for the DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    return summary

def clean_data_pipeline(df, config):
    """
    Execute a complete data cleaning pipeline based on configuration.
    
    Args:
        df: pandas DataFrame to clean
        config: dictionary containing cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    if not validate_dataframe(df):
        return df
    
    # Remove duplicates if configured
    if config.get('remove_duplicates', False):
        subset = config.get('duplicate_subset', None)
        df = remove_duplicates(df, subset=subset)
    
    # Clean numeric columns if specified
    numeric_cols = config.get('numeric_columns', [])
    if numeric_cols:
        df = clean_numeric_columns(df, numeric_cols)
    
    # Drop columns if specified
    drop_cols = config.get('drop_columns', [])
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    
    # Reset index if configured
    if config.get('reset_index', False):
        df = df.reset_index(drop=True)
    
    return df