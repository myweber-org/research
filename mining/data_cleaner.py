
import pandas as pd

def clean_dataframe(df, fill_strategy='drop', column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    fill_strategy (str): Strategy for handling null values. 
                         Options: 'drop' to remove rows, 'fill' to fill with column mean (numeric) or mode (object).
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
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['errors'].append('DataFrame is empty')
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Missing required columns: {missing_columns}')
    
    return validation_result

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {'Name': ['Alice', 'Bob', None], 'Age': [25, None, 30], 'Score': [85.5, 92.0, 88.5]}
#     df = pd.DataFrame(sample_data)
#     
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned_df = clean_dataframe(df, fill_strategy='fill', column_case='title')
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)
#     
#     validation = validate_dataframe(cleaned_df, required_columns=['Name', 'Age', 'Score'])
#     print("\nValidation Result:")
#     print(validation)
import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def remove_outliers(data, column, threshold=1.5):
    """
    Remove outliers from specified column
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_column(data, column, method='minmax'):
    """
    Normalize column using specified method
    """
    if method == 'minmax':
        min_val = data[column].min()
        max_val = data[column].max()
        data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = data[column].mean()
        std_val = data[column].std()
        data[column + '_normalized'] = (data[column] - mean_val) / std_val
    
    return data

def handle_missing_values(data, column, strategy='mean'):
    """
    Handle missing values in specified column
    """
    if strategy == 'mean':
        fill_value = data[column].mean()
    elif strategy == 'median':
        fill_value = data[column].median()
    elif strategy == 'mode':
        fill_value = data[column].mode()[0]
    else:
        fill_value = 0
    
    data[column] = data[column].fillna(fill_value)
    return data

def clean_dataset(data, numeric_columns, outlier_threshold=1.5, normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = handle_missing_values(cleaned_data, column)
            cleaned_data = remove_outliers(cleaned_data, column, outlier_threshold)
            cleaned_data = normalize_column(cleaned_data, column, normalize_method)
    
    return cleaned_data