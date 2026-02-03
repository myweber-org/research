
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    mean = df[column].mean()
    std = df[column].std()
    df[column] = (df[column] - mean) / std
    return df

def clean_dataset(input_file, output_file):
    df = load_data(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        data: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    for col in columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            col_min = normalized_data[col].min()
            col_max = normalized_data[col].max()
            if col_max != col_min:
                normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
            else:
                normalized_data[col] = 0
    
    return normalized_data

def standardize_zscore(data, columns=None):
    """
    Standardize specified columns using z-score normalization.
    
    Args:
        data: pandas DataFrame
        columns: list of column names to standardize (default: all numeric columns)
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    standardized_data = data.copy()
    for col in columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            col_mean = standardized_data[col].mean()
            col_std = standardized_data[col].std()
            if col_std > 0:
                standardized_data[col] = (standardized_data[col] - col_mean) / col_std
            else:
                standardized_data[col] = 0
    
    return standardized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', or 'drop')
        columns: list of column names to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = data.columns
    
    processed_data = data.copy()
    
    for col in columns:
        if col not in processed_data.columns:
            continue
            
        if processed_data[col].isnull().any():
            if strategy == 'drop':
                processed_data = processed_data.dropna(subset=[col])
            elif strategy == 'mean' and np.issubdtype(processed_data[col].dtype, np.number):
                processed_data[col].fillna(processed_data[col].mean(), inplace=True)
            elif strategy == 'median' and np.issubdtype(processed_data[col].dtype, np.number):
                processed_data[col].fillna(processed_data[col].median(), inplace=True)
            elif strategy == 'mode':
                processed_data[col].fillna(processed_data[col].mode()[0], inplace=True)
    
    return processed_data

def clean_dataset(data, config):
    """
    Apply multiple cleaning operations based on configuration.
    
    Args:
        data: pandas DataFrame
        config: dictionary with cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if 'missing_values' in config:
        cleaned_data = handle_missing_values(
            cleaned_data,
            strategy=config['missing_values'].get('strategy', 'mean'),
            columns=config['missing_values'].get('columns')
        )
    
    if 'outliers' in config:
        for col_config in config['outliers']:
            cleaned_data = remove_outliers_iqr(
                cleaned_data,
                column=col_config['column'],
                multiplier=col_config.get('multiplier', 1.5)
            )
    
    if 'normalization' in config:
        norm_type = config['normalization'].get('type', 'minmax')
        columns = config['normalization'].get('columns')
        
        if norm_type == 'minmax':
            cleaned_data = normalize_minmax(cleaned_data, columns)
        elif norm_type == 'zscore':
            cleaned_data = standardize_zscore(cleaned_data, columns)
    
    return cleaned_data