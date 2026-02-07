
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def z_score_normalize(dataframe, columns):
    """
    Apply z-score normalization to specified columns
    """
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        
        mean_val = normalized_df[col].mean()
        std_val = normalized_df[col].std()
        
        if std_val > 0:
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        else:
            normalized_df[col] = 0
    
    return normalized_df

def min_max_normalize(dataframe, columns, feature_range=(0, 1)):
    """
    Apply min-max normalization to specified columns
    """
    normalized_df = dataframe.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        
        if col_max > col_min:
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            normalized_df[col] = normalized_df[col] * (max_val - min_val) + min_val
        else:
            normalized_df[col] = min_val
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy
    """
    processed_df = dataframe.copy()
    
    if columns is None:
        columns = processed_df.columns
    
    for col in columns:
        if col not in processed_df.columns:
            continue
            
        if processed_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = processed_df[col].mean()
            elif strategy == 'median':
                fill_value = processed_df[col].median()
            elif strategy == 'mode':
                fill_value = processed_df[col].mode()[0]
            elif strategy == 'drop':
                processed_df = processed_df.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            processed_df[col] = processed_df[col].fillna(fill_value)
    
    return processed_df

def create_data_summary(dataframe):
    """
    Create comprehensive data summary
    """
    summary = {
        'total_rows': len(dataframe),
        'total_columns': len(dataframe.columns),
        'missing_values': dataframe.isnull().sum().sum(),
        'duplicate_rows': dataframe.duplicated().sum(),
        'numeric_columns': list(dataframe.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(dataframe.select_dtypes(include=['object']).columns),
        'column_types': dataframe.dtypes.to_dict()
    }
    
    return summarydef remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def clean_numeric_data(values, default=0):
    """
    Convert all items in a list to integers, replacing non-convertible items with a default value.
    """
    cleaned = []
    for value in values:
        try:
            cleaned.append(int(value))
        except (ValueError, TypeError):
            cleaned.append(default)
    return cleaned

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))
    
    numeric_mix = [10, "20", "abc", 30.5, None]
    print("Numeric original:", numeric_mix)
    print("Numeric cleaned:", clean_numeric_data(numeric_mix))import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list, optional): Column labels to consider for duplicates.
    keep (str, optional): Which duplicates to keep.
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean a numeric column by handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of the column to clean.
    fill_method (str): Method to fill missing values ('mean', 'median', 'zero').
    
    Returns:
    pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df_clean = df.copy()
    
    if fill_method == 'mean':
        fill_value = df_clean[column_name].mean()
    elif fill_method == 'median':
        fill_value = df_clean[column_name].median()
    elif fill_method == 'zero':
        fill_value = 0
    else:
        raise ValueError("fill_method must be 'mean', 'median', or 'zero'")
    
    df_clean[column_name] = df_clean[column_name].fillna(fill_value)
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
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