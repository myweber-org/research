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