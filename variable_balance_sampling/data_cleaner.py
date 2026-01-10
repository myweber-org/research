import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        Cleaned DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in DataFrame columns.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: list of columns to fill, None for all numeric columns
    
    Returns:
        DataFrame with missing values filled
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    
    for col in columns:
        if col in df.columns:
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_filled[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'constant':
                df_filled[col] = df[col].fillna(0)
    
    return df_filled

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to normalize, None for all numeric columns
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if col in df.columns:
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val != 0:
                    df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def clean_dataframe(df, remove_dups=True, fill_na=True, normalize=True):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df: pandas DataFrame
        remove_dups: whether to remove duplicates
        fill_na: whether to fill missing values
        normalize: whether to normalize numeric columns
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df)
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns boolean mask for outliers.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, columns, threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    """
    clean_data = data.copy()
    for col in columns:
        outliers = detect_outliers_iqr(clean_data, col, threshold)
        clean_data = clean_data[~outliers]
    return clean_data.reset_index(drop=True)

def normalize_minmax(data, columns):
    """
    Apply min-max normalization to specified columns.
    """
    normalized_data = data.copy()
    for col in columns:
        min_val = normalized_data[col].min()
        max_val = normalized_data[col].max()
        normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
    return normalized_data

def standardize_zscore(data, columns):
    """
    Apply z-score standardization to specified columns.
    """
    standardized_data = data.copy()
    for col in columns:
        mean_val = standardized_data[col].mean()
        std_val = standardized_data[col].std()
        standardized_data[col] = (standardized_data[col] - mean_val) / std_val
    return standardized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    """
    if columns is None:
        columns = data.columns
    
    filled_data = data.copy()
    
    for col in columns:
        if filled_data[col].isnull().any():
            if strategy == 'mean':
                fill_value = filled_data[col].mean()
            elif strategy == 'median':
                fill_value = filled_data[col].median()
            elif strategy == 'mode':
                fill_value = filled_data[col].mode()[0]
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            filled_data[col] = filled_data[col].fillna(fill_value)
    
    return filled_data

def clean_dataset(data, numeric_columns, outlier_threshold=1.5, 
                  normalization='standardize', missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    """
    # Handle missing values
    cleaned_data = handle_missing_values(data, strategy=missing_strategy, 
                                         columns=numeric_columns)
    
    # Remove outliers
    cleaned_data = remove_outliers(cleaned_data, numeric_columns, 
                                   threshold=outlier_threshold)
    
    # Apply normalization/standardization
    if normalization == 'minmax':
        cleaned_data = normalize_minmax(cleaned_data, numeric_columns)
    elif normalization == 'standardize':
        cleaned_data = standardize_zscore(cleaned_data, numeric_columns)
    
    return cleaned_data

def validate_data(data, numeric_columns):
    """
    Validate cleaned data by checking statistics.
    """
    validation_report = {}
    
    for col in numeric_columns:
        validation_report[col] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'missing_values': data[col].isnull().sum(),
            'zero_values': (data[col] == 0).sum()
        }
    
    return pd.DataFrame(validation_report).T

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000)
    })
    
    # Add some outliers and missing values
    sample_data.loc[50:55, 'feature_a'] = 500
    sample_data.loc[100:105, 'feature_b'] = -100
    sample_data.loc[200:205, 'feature_c'] = np.nan
    
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    
    # Clean the data
    cleaned = clean_dataset(sample_data, numeric_cols)
    
    # Validate results
    validation = validate_data(cleaned, numeric_cols)
    print("Data validation report:")
    print(validation)