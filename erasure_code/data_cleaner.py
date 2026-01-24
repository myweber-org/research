
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    return stats

def clean_dataset(df, numeric_columns):
    """
    Clean multiple numeric columns in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_dfimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas Series using the IQR method.
    Returns a cleaned Series and the indices of outliers removed.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series")
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    cleaned_data = data[~outlier_mask].copy()
    
    return cleaned_data, data.index[outlier_mask].tolist()

def normalize_minmax(data):
    """
    Normalize data to [0, 1] range using min-max scaling.
    Handles NaN values by ignoring them in calculation.
    """
    if isinstance(data, pd.Series):
        data_values = data.values
    elif isinstance(data, np.ndarray):
        data_values = data
    else:
        raise TypeError("Input must be pandas Series or numpy array")
    
    valid_mask = ~np.isnan(data_values)
    if not np.any(valid_mask):
        return np.full_like(data_values, np.nan)
    
    valid_data = data_values[valid_mask]
    data_min = np.min(valid_data)
    data_max = np.max(valid_data)
    
    if data_max == data_min:
        normalized = np.zeros_like(data_values)
    else:
        normalized = (data_values - data_min) / (data_max - data_min)
    
    normalized[~valid_mask] = np.nan
    return normalized

def winsorize_data(data, limits=(0.05, 0.05)):
    """
    Apply winsorization to limit extreme values.
    Returns winsorized data with extreme values replaced.
    """
    if isinstance(data, pd.Series):
        result = stats.mstats.winsorize(data.values, limits=limits)
        return pd.Series(result, index=data.index)
    elif isinstance(data, np.ndarray):
        return stats.mstats.winsorize(data, limits=limits)
    else:
        raise TypeError("Input must be pandas Series or numpy array")

def clean_dataframe(df, numeric_columns=None, method='iqr'):
    """
    Clean a DataFrame by removing outliers from numeric columns.
    Supports 'iqr' and 'winsorize' methods.
    Returns cleaned DataFrame and outlier report dictionary.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    outlier_report = {}
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        original_data = df[col]
        
        if method == 'iqr':
            cleaned_series, outliers = remove_outliers_iqr(original_data, col)
            cleaned_df.loc[cleaned_series.index, col] = cleaned_series
            outlier_report[col] = {
                'outlier_indices': outliers,
                'outlier_count': len(outliers),
                'method': 'iqr'
            }
        elif method == 'winsorize':
            winsorized = winsorize_data(original_data)
            cleaned_df[col] = winsorized
            outlier_report[col] = {
                'method': 'winsorize',
                'limits': (0.05, 0.05)
            }
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'winsorize'")
    
    return cleaned_df, outlier_report

def process_dataset(filepath, output_path=None):
    """
    Complete pipeline to load, clean, and optionally save a dataset.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise IOError(f"Failed to read file {filepath}: {str(e)}")
    
    print(f"Loaded dataset with shape: {df.shape}")
    
    cleaned_df, report = clean_dataframe(df)
    
    print("Cleaning completed. Outlier report:")
    for col, info in report.items():
        if 'outlier_count' in info:
            print(f"  {col}: Removed {info['outlier_count']} outliers")
    
    if output_path:
        cleaned_df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return cleaned_df, report