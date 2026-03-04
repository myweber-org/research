
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Strategy to fill missing values. 
                       Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                else:  # mode
                    fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value:.2f}")
    
    print(f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df.empty:
        print("Validation failed: DataFrame is empty.")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed.")
    return True

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to check for outliers. 
                   If None, uses all numeric columns.
    multiplier (float): IQR multiplier for outlier detection. Default is 1.5.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    original_len = len(df)
    mask = pd.Series([True] * len(df), index=df.index)
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            mask = mask & col_mask
    
    cleaned_df = df[mask].copy()
    outliers_removed = original_len - len(cleaned_df)
    
    if outliers_removed > 0:
        print(f"Removed {outliers_removed} outliers using IQR method.")
    
    return cleaned_df
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_dfimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column]
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 50),
        'feature_b': np.random.exponential(2, 50),
        'feature_c': np.random.randint(1, 100, 50)
    })
    
    print("Original data shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b', 'feature_c'])
    print("Cleaned data shape:", cleaned.shape)
    print("Data validation passed:", validate_data(cleaned, ['feature_a', 'feature_b', 'feature_c']))
import pandas as pd

def clean_dataset(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): If True, drop rows with any null values.
        rename_columns (bool): If True, rename columns to lowercase with underscores.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df.columns = (
            cleaned_df.columns
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^a-z0-9_]', '', regex=True)
        )
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
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

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list, optional): Columns to consider for duplicates.
    keep (str, optional): Which duplicates to keep.
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by filling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of column to clean.
    fill_method (str): Method to fill missing values.
    
    Returns:
    pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if df[column_name].dtype not in ['int64', 'float64']:
        raise ValueError(f"Column '{column_name}' is not numeric")
    
    df_clean = df.copy()
    
    if fill_method == 'mean':
        fill_value = df[column_name].mean()
    elif fill_method == 'median':
        fill_value = df[column_name].median()
    elif fill_method == 'mode':
        fill_value = df[column_name].mode()[0]
    else:
        fill_value = 0
    
    df_clean[column_name] = df_clean[column_name].fillna(fill_value)
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of required columns.
    
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
    
    return True, "DataFrame is valid"

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 3, 4, 4, 5],
        'value': [10.5, None, 15.2, 20.1, 20.1, None],
        'category': ['A', 'B', 'A', 'C', 'C', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    df_no_dups = remove_duplicates(df, subset=['id'], keep='first')
    print("DataFrame after removing duplicate IDs:")
    print(df_no_dups)
    print()
    
    df_clean = clean_numeric_column(df_no_dups, 'value', fill_method='mean')
    print("DataFrame after cleaning numeric column:")
    print(df_clean)
    print()
    
    is_valid, message = validate_dataframe(df_clean, required_columns=['id', 'value'])
    print(f"DataFrame validation: {is_valid} - {message}")

if __name__ == "__main__":
    main()