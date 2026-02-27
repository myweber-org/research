import numpy as np
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
    data_array = np.array(data, dtype=float)
    valid_mask = ~np.isnan(data_array)
    
    if not np.any(valid_mask):
        return np.full_like(data_array, np.nan)
    
    valid_data = data_array[valid_mask]
    data_min = np.min(valid_data)
    data_max = np.max(valid_data)
    
    if data_max == data_min:
        normalized = np.zeros_like(data_array)
    else:
        normalized = (data_array - data_min) / (data_max - data_min)
    
    normalized[~valid_mask] = np.nan
    return normalized

def winsorize_data(data, limits=(0.05, 0.05)):
    """
    Apply winsorization to limit extreme values.
    Returns winsorized data preserving original shape.
    """
    try:
        winsorized = stats.mstats.winsorize(data, limits=limits)
        return winsorized.data if hasattr(winsorized, 'data') else winsorized
    except Exception as e:
        print(f"Winsorization failed: {e}")
        return data

def clean_dataframe(df, numeric_columns=None, method='iqr'):
    """
    Clean a DataFrame by handling outliers in numeric columns.
    Supports 'iqr' or 'winsorize' methods.
    Returns cleaned DataFrame and outlier report.
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
                'method': 'iqr',
                'outlier_count': len(outliers),
                'outlier_indices': outliers
            }
        elif method == 'winsorize':
            winsorized = winsorize_data(original_data.values)
            cleaned_df[col] = winsorized
            outlier_report[col] = {
                'method': 'winsorize',
                'limits': (0.05, 0.05)
            }
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'winsorize'")
    
    return cleaned_df, outlier_report

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 50),
        'B': np.random.exponential(2, 50),
        'C': np.random.uniform(0, 1, 50)
    })
    
    # Add some outliers
    sample_data.loc[5, 'A'] = 500
    sample_data.loc[10, 'B'] = 50
    
    print("Original data shape:", sample_data.shape)
    print("Original data summary:")
    print(sample_data.describe())
    
    cleaned, report = clean_dataframe(sample_data, method='iqr')
    print("\nCleaned data shape:", cleaned.shape)
    print("Outlier report:", report)
    
    normalized_col = normalize_minmax(sample_data['A'])
    print(f"\nNormalized column 'A' range: [{np.nanmin(normalized_col):.3f}, {np.nanmax(normalized_col):.3f}]")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to analyze.
    
    Returns:
        dict: Dictionary containing statistics.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
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
    Clean dataset by removing outliers from multiple numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_columns (list): List of numeric column names to clean.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.randn(100) * 10 + 50
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset shape:", df.shape)
    
    cleaned_df = clean_dataset(df, ['value'])
    print("Cleaned dataset shape:", cleaned_df.shape)
    
    stats = calculate_basic_stats(cleaned_df, 'value')
    print("\nBasic statistics after cleaning:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")