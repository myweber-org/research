def remove_duplicates(data_list):
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

def clean_numeric_strings(data_list):
    """
    Convert string representations of numbers to integers where possible.
    Non-convertible items remain unchanged.
    """
    cleaned = []
    for item in data_list:
        if isinstance(item, str) and item.isdigit():
            cleaned.append(int(item))
        else:
            cleaned.append(item)
    return cleaned

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, "4", "4", "five", 3, 1]
    print("Original:", sample_data)
    
    unique_data = remove_duplicates(sample_data)
    print("After deduplication:", unique_data)
    
    cleaned_data = clean_numeric_strings(unique_data)
    print("After numeric cleaning:", cleaned_data)import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data (np.ndarray): Input data array
        column (int): Index of column to clean
    
    Returns:
        np.ndarray: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    column_data = data[:, column]
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the cleaned data.
    
    Args:
        data (np.ndarray): Input data array
    
    Returns:
        dict: Dictionary containing mean, median, and std
    """
    if data.size == 0:
        return {"mean": 0, "median": 0, "std": 0}
    
    return {
        "mean": np.mean(data, axis=0),
        "median": np.median(data, axis=0),
        "std": np.std(data, axis=0)
    }

def validate_data(data):
    """
    Validate input data for cleaning operations.
    
    Args:
        data: Input data to validate
    
    Returns:
        bool: True if data is valid, False otherwise
    """
    if data is None:
        return False
    
    if isinstance(data, np.ndarray):
        return data.size > 0
    
    try:
        arr = np.array(data)
        return arr.size > 0
    except:
        return False

def process_dataset(data, columns_to_clean):
    """
    Process dataset by cleaning multiple columns.
    
    Args:
        data (np.ndarray): Input data array
        columns_to_clean (list): List of column indices to clean
    
    Returns:
        tuple: (cleaned_data, statistics_dict)
    """
    if not validate_data(data):
        raise ValueError("Invalid input data")
    
    cleaned_data = data.copy()
    
    for column in columns_to_clean:
        if 0 <= column < cleaned_data.shape[1]:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    stats = calculate_statistics(cleaned_data)
    
    return cleaned_data, stats
import pandas as pd
import numpy as np

def remove_missing_values(df, threshold=0.5):
    """
    Remove columns with missing values exceeding threshold percentage.
    """
    missing_percent = df.isnull().sum() / len(df)
    columns_to_drop = missing_percent[missing_percent > threshold].index
    return df.drop(columns=columns_to_drop)

def fill_missing_with_median(df, columns=None):
    """
    Fill missing values with column median.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            df_filled[col] = df[col].fillna(df[col].median())
    return df_filled

def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def cap_outliers(df, column, method='iqr'):
    """
    Cap outliers to specified bounds.
    """
    df_capped = df.copy()
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    elif method == 'percentile':
        lower_bound = df[column].quantile(0.01)
        upper_bound = df[column].quantile(0.99)
    else:
        raise ValueError("Method must be 'iqr' or 'percentile'")
    
    df_capped[column] = df_capped[column].clip(lower=lower_bound, upper=upper_bound)
    return df_capped

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to zero mean and unit variance.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    for col in columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df_standardized[col] = (df[col] - mean) / std
    return df_standardized

def clean_dataset(df, missing_threshold=0.5, outlier_method='iqr'):
    """
    Comprehensive data cleaning pipeline.
    """
    df_clean = remove_missing_values(df, threshold=missing_threshold)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean = fill_missing_with_median(df_clean, columns=numeric_cols)
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean = cap_outliers(df_clean, col, method=outlier_method)
    
    df_clean = standardize_columns(df_clean, columns=numeric_cols)
    return df_clean