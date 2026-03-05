import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10, 20, 20, None, 40],
        'category': ['A', 'B', 'B', 'C', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df)
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, ['id', 'value'])
    print(f"\nValidation: {is_valid} - {message}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'")

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='minmax', missing_strategy='mean'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if missing_strategy:
            cleaned_df = handle_missing_values(cleaned_df[[col]], strategy=missing_strategy)
        
        if outlier_removal:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        
        if normalization == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalization == 'zscore':
            cleaned_df = standardize_zscore(cleaned_df, col)
    
    return cleaned_df
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_standard(data, column):
    """
    Normalize data using Standard scaling
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    normalized = (data[column] - mean_val) / std_val
    return normalized

def clean_missing_values(data, strategy='mean'):
    """
    Handle missing values with different strategies
    """
    cleaned_data = data.copy()
    
    for column in cleaned_data.columns:
        if cleaned_data[column].isnull().any():
            if strategy == 'mean':
                cleaned_data[column].fillna(cleaned_data[column].mean(), inplace=True)
            elif strategy == 'median':
                cleaned_data[column].fillna(cleaned_data[column].median(), inplace=True)
            elif strategy == 'mode':
                cleaned_data[column].fillna(cleaned_data[column].mode()[0], inplace=True)
            elif strategy == 'drop':
                cleaned_data.dropna(subset=[column], inplace=True)
    
    return cleaned_data

def validate_dataframe(data):
    """
    Validate dataframe structure and content
    """
    validation_results = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'duplicate_rows': data.duplicated().sum(),
        'data_types': data.dtypes.to_dict()
    }
    return validation_results

def process_data_pipeline(data, config):
    """
    Complete data processing pipeline
    """
    processed_data = data.copy()
    
    # Clean missing values
    if 'missing_strategy' in config:
        processed_data = clean_missing_values(processed_data, config['missing_strategy'])
    
    # Handle outliers
    if 'outlier_columns' in config:
        for column in config['outlier_columns']:
            if column in processed_data.columns:
                processed_data = remove_outliers_zscore(processed_data, column)
    
    # Normalize data
    if 'normalize_columns' in config:
        for column, method in config['normalize_columns'].items():
            if column in processed_data.columns:
                if method == 'minmax':
                    processed_data[column] = normalize_minmax(processed_data, column)
                elif method == 'standard':
                    processed_data[column] = normalize_standard(processed_data, column)
    
    return processed_data