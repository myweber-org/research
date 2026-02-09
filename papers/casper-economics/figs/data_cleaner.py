
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

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
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
    
    return True, "DataFrame is valid"import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and basic integrity.
    
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

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid}, Message: {message}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns):
    """
    Remove outliers using IQR method for specified columns.
    Returns cleaned DataFrame and outlier indices.
    """
    cleaned_df = df.copy()
    outlier_indices = []
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = cleaned_df[(cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)]
        outlier_indices.extend(outliers.index.tolist())
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    
    return cleaned_df, list(set(outlier_indices))

def normalize_minmax(df, columns):
    """
    Normalize specified columns using min-max scaling.
    Returns DataFrame with normalized columns.
    """
    normalized_df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        
        if max_val > min_val:
            normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    
    return normalized_df

def detect_skewed_columns(df, threshold=0.5):
    """
    Detect columns with skewed distributions.
    Returns list of column names with skewness above threshold.
    """
    skewed_columns = []
    
    for col in df.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(df[col].dropna())
        if abs(skewness) > threshold:
            skewed_columns.append((col, skewness))
    
    return sorted(skewed_columns, key=lambda x: abs(x[1]), reverse=True)

def log_transform_skewed(df, skewed_columns):
    """
    Apply log transformation to skewed columns.
    Handles zero and negative values by adding constant.
    """
    transformed_df = df.copy()
    
    for col, _ in skewed_columns:
        if col not in df.columns:
            continue
            
        col_data = transformed_df[col].copy()
        min_val = col_data.min()
        
        if min_val <= 0:
            constant = abs(min_val) + 1
            transformed_df[col] = np.log(col_data + constant)
        else:
            transformed_df[col] = np.log(col_data)
    
    return transformed_df

def clean_dataset(df, numeric_columns=None, remove_outliers=True, normalize=True, handle_skew=True):
    """
    Main function to clean dataset with multiple preprocessing steps.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    if remove_outliers:
        cleaned_df, _ = remove_outliers_iqr(cleaned_df, numeric_columns)
    
    if handle_skew:
        skewed_cols = detect_skewed_columns(cleaned_df[numeric_columns])
        if skewed_cols:
            cleaned_df = log_transform_skewed(cleaned_df, skewed_cols)
    
    if normalize:
        cleaned_df = normalize_minmax(cleaned_df, numeric_columns)
    
    return cleaned_df