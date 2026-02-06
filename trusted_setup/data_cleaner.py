
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
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
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

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    removal_stats = {}
    
    for col in numeric_columns:
        if col not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data, removed = remove_outliers_iqr(cleaned_data, col)
        elif outlier_method == 'zscore':
            cleaned_data, removed = remove_outliers_zscore(cleaned_data, col)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
        
        removal_stats[col] = removed
        
        if normalize_method == 'minmax':
            cleaned_data[col] = normalize_minmax(cleaned_data, col)
        elif normalize_method == 'zscore':
            cleaned_data[col] = normalize_zscore(cleaned_data, col)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
    
    return cleaned_data, removal_stats

def validate_data(data, required_columns, numeric_columns):
    """
    Validate dataset structure and content
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    for col in numeric_columns:
        if col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column '{col}' must be numeric")
    
    return True
import pandas as pd
import re

def clean_dataframe(df, columns_to_normalize=None, remove_duplicates=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing specified columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_normalize (list): List of column names to normalize strings
    remove_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if columns_to_normalize:
        for col in columns_to_normalize:
            if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].apply(normalize_string)
                print(f"Normalized column: {col}")
    
    return cleaned_df

def normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Parameters:
    text (str): Input string to normalize
    
    Returns:
    str: Normalized string
    """
    if not isinstance(text, str):
        return text
    
    # Convert to lowercase
    normalized = text.lower()
    
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    
    # Remove special characters except alphanumeric and spaces
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    return normalized

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    email_column (str): Name of the column containing email addresses
    
    Returns:
    pd.DataFrame: DataFrame with validation results
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validation_results = df.copy()
    validation_results['email_valid'] = validation_results[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    valid_count = validation_results['email_valid'].sum()
    total_count = len(validation_results)
    
    print(f"Email validation results: {valid_count}/{total_count} valid emails")
    
    return validation_results
def deduplicate_list(original_list):
    seen = set()
    deduplicated = []
    for item in original_list:
        if item not in seen:
            seen.add(item)
            deduplicated.append(item)
    return deduplicated