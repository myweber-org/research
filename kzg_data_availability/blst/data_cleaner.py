
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a column using the IQR method.
    
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
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column values to range [0, 1] using min-max scaling.
    
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
    Standardize column values using z-score normalization.
    
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

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        DataFrame with missing values handled
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return data.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data[col] = data[col].fillna(fill_value)
    
    return data

def create_cleaning_pipeline(data, config):
    """
    Apply multiple cleaning operations based on configuration.
    
    Args:
        data: pandas DataFrame
        config: dictionary with cleaning operations and parameters
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if 'remove_outliers' in config:
        for col in config['remove_outliers'].get('columns', []):
            factor = config['remove_outliers'].get('factor', 1.5)
            cleaned_data = remove_outliers_iqr(cleaned_data, col, factor)
    
    if 'normalize' in config:
        for col in config['normalize'].get('columns', []):
            method = config['normalize'].get('method', 'minmax')
            if method == 'minmax':
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
            elif method == 'zscore':
                cleaned_data[col] = standardize_zscore(cleaned_data, col)
    
    if 'handle_missing' in config:
        strategy = config['handle_missing'].get('strategy', 'mean')
        cleaned_data = handle_missing_values(cleaned_data, strategy)
    
    return cleaned_dataimport pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str or None): Method to fill missing values:
        - 'mean': Fill with column mean (numeric only)
        - 'median': Fill with column median (numeric only)
        - 'mode': Fill with column mode
        - None: Do not fill missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        if fill_missing == 'mean':
            for col in numeric_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        elif fill_missing == 'median':
            for col in numeric_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
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

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'A': [1, 2, 2, None, 5],
#         'B': [10, None, 30, 40, 50],
#         'C': ['x', 'y', 'x', 'z', None]
#     }
#     df = pd.DataFrame(sample_data)
#     cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame:")
#     print(cleaned)import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing values in a DataFrame using specified strategy.
    
    Args:
        df: pandas DataFrame containing data with missing values
        strategy: Method for handling missing values ('mean', 'median', 'mode', 'drop')
        columns: List of columns to apply cleaning to (None for all columns)
    
    Returns:
        Cleaned pandas DataFrame
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].isnull().sum() == 0:
            continue
        
        if strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        elif strategy == 'mode':
            if not df_clean[col].empty:
                mode_value = df_clean[col].mode()
                if not mode_value.empty:
                    df_clean[col].fillna(mode_value[0], inplace=True)
        
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
    
    return df_clean

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        column: Column name to check for outliers
        threshold: IQR multiplier for outlier detection
    
    Returns:
        Boolean mask indicating outliers
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return pd.Series([False] * len(df), index=df.index)
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df: pandas DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        Series with normalized values
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return df[column]
    
    if method == 'minmax':
        col_min = df[column].min()
        col_max = df[column].max()
        
        if col_max == col_min:
            return pd.Series([0.5] * len(df), index=df.index)
        
        return (df[column] - col_min) / (col_max - col_min)
    
    elif method == 'zscore':
        col_mean = df[column].mean()
        col_std = df[column].std()
        
        if col_std == 0:
            return pd.Series([0] * len(df), index=df.index)
        
        return (df[column] - col_mean) / col_std
    
    return df[column]

if __name__ == "__main__":
    sample_data = {
        'age': [25, 30, np.nan, 35, 40, 150, 28],
        'salary': [50000, 60000, 70000, np.nan, 90000, 100000, 55000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Finance']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_missing_data(df, strategy='mean')
    print("DataFrame after cleaning missing values:")
    print(cleaned_df)
    print("\n")
    
    outliers = detect_outliers_iqr(cleaned_df, 'age')
    print("Outliers in 'age' column:")
    print(cleaned_df[outliers])
    print("\n")
    
    cleaned_df['age_normalized'] = normalize_column(cleaned_df, 'age', method='minmax')
    print("DataFrame with normalized age column:")
    print(cleaned_df)