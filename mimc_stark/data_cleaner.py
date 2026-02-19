
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
    
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
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
    
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Remove outliers
    if outlier_method == 'iqr':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
    elif outlier_method == 'zscore':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    # Normalize data
    if normalize_method == 'minmax':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
    elif normalize_method == 'zscore':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate statistical summary of the data.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        Dictionary with statistical summary
    """
    summary = {
        'shape': data.shape,
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_summary': {}
    }
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_summary'][col] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            '25%': data[col].quantile(0.25),
            '50%': data[col].quantile(0.50),
            '75%': data[col].quantile(0.75),
            'max': data[col].max(),
            'skewness': data[col].skew(),
            'kurtosis': data[col].kurtosis()
        }
    
    return summaryimport pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicate rows and handling missing values.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to clean.
    columns_to_check (list, optional): List of column names to check for duplicates.
                                       If None, checks all columns.
    fill_missing (str or value, optional): Method to fill missing values.
                                           Can be 'mean', 'median', 'mode', or a scalar value.
                                           Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if columns_to_check is None:
        cleaned_df = cleaned_df.drop_duplicates()
    else:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    
    # Handle missing values
    for column in cleaned_df.select_dtypes(include=['number']).columns:
        if cleaned_df[column].isnull().any():
            if fill_missing == 'mean':
                fill_value = cleaned_df[column].mean()
            elif fill_missing == 'median':
                fill_value = cleaned_df[column].median()
            elif fill_missing == 'mode':
                fill_value = cleaned_df[column].mode()[0]
            else:
                fill_value = fill_missing
            
            cleaned_df[column] = cleaned_df[column].fillna(fill_value)
    
    # For non-numeric columns, fill with the most frequent value
    for column in cleaned_df.select_dtypes(exclude=['number']).columns:
        if cleaned_df[column].isnull().any():
            mode_value = cleaned_df[column].mode()
            if not mode_value.empty:
                cleaned_df[column] = cleaned_df[column].fillna(mode_value[0])
            else:
                cleaned_df[column] = cleaned_df[column].fillna('Unknown')
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame by checking for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to validate.
    required_columns (list, optional): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    dict: Dictionary containing validation results and error messages.
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check for required columns
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for empty DataFrame
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['errors'].append("DataFrame is empty")
    
    # Check for all NaN columns
    all_nan_columns = df.columns[df.isnull().all()].tolist()
    if all_nan_columns:
        validation_result['warnings'].append(f"Columns with all NaN values: {all_nan_columns}")
    
    return validation_result['is_valid'], validation_result

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4, 5],
#         'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
#         'age': [25, 30, 30, None, 35, 40],
#         'score': [85.5, 92.0, 92.0, 78.5, None, 95.0]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\n")
#     
#     # Clean the dataset
#     cleaned = clean_dataset(df, columns_to_check=['id', 'name'], fill_missing='mean')
#     print("Cleaned DataFrame:")
#     print(cleaned)
#     print("\n")
#     
#     # Validate the dataset
#     is_valid, validation_info = validate_dataset(cleaned, required_columns=['id', 'name', 'age'])
#     print(f"Validation passed: {is_valid}")
#     print(f"Validation details: {validation_info}")