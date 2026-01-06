
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. 
                        Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            elif fill_missing == 'mode':
                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'x', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid} - {message}")
def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport numpy as np
import pandas as pd

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
    outlier_indices = data[outlier_mask].index.tolist()
    
    return cleaned_data, outlier_indices

def normalize_minmax(data):
    """
    Normalize data to [0, 1] range using min-max scaling.
    Handles both pandas Series and numpy arrays.
    """
    if isinstance(data, pd.Series):
        data_values = data.values
    elif isinstance(data, np.ndarray):
        data_values = data
    else:
        raise TypeError("Data must be pandas Series or numpy array")
    
    if len(data_values) == 0:
        return data_values
    
    data_min = np.min(data_values)
    data_max = np.max(data_values)
    
    if data_max == data_min:
        return np.zeros_like(data_values)
    
    normalized = (data_values - data_min) / (data_max - data_min)
    
    if isinstance(data, pd.Series):
        return pd.Series(normalized, index=data.index)
    return normalized

def clean_dataframe(df, numeric_columns=None):
    """
    Clean a DataFrame by removing outliers from numeric columns
    and normalizing them. Returns cleaned DataFrame and outlier report.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    outlier_report = {}
    
    for col in numeric_columns:
        if col in df.columns:
            original_series = df[col].dropna()
            if len(original_series) > 0:
                cleaned_series, outliers = remove_outliers_iqr(original_series, col)
                cleaned_df.loc[cleaned_series.index, col] = cleaned_series
                
                if outliers:
                    outlier_report[col] = {
                        'count': len(outliers),
                        'indices': outliers,
                        'values': df.loc[outliers, col].tolist()
                    }
                
                normalized_series = normalize_minmax(cleaned_series)
                cleaned_df.loc[normalized_series.index, col] = normalized_series
    
    return cleaned_df, outlier_report

def validate_data_quality(df, threshold=0.1):
    """
    Validate data quality by checking for missing values and data ranges.
    Returns quality report dictionary.
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'data_types': {},
        'numeric_ranges': {}
    }
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_percentage = missing_count / len(df)
        report['missing_values'][col] = {
            'count': missing_count,
            'percentage': missing_percentage,
            'is_acceptable': missing_percentage <= threshold
        }
        
        report['data_types'][col] = str(df[col].dtype)
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_data = df[col].dropna()
            if len(col_data) > 0:
                report['numeric_ranges'][col] = {
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std())
                }
    
    return report

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['X', 'Y', 'Z'], 1000)
    })
    
    sample_data.loc[np.random.choice(1000, 50), 'A'] = np.random.uniform(500, 1000, 50)
    sample_data.loc[np.random.choice(1000, 30), 'B'] = np.random.uniform(500, 1000, 30)
    
    print("Original data shape:", sample_data.shape)
    
    quality_report = validate_data_quality(sample_data)
    print("\nData quality report:")
    for col, info in quality_report['missing_values'].items():
        print(f"{col}: {info['count']} missing values ({info['percentage']:.1%})")
    
    cleaned_df, outliers = clean_dataframe(sample_data, ['A', 'B', 'C'])
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    
    print("\nOutliers removed:")
    for col, info in outliers.items():
        print(f"{col}: {info['count']} outliers")