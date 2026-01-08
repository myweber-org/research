
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_na=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling null values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_na (bool): Whether to fill null values
    fill_value: Value to use for filling nulls
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_na:
        null_count = cleaned_df.isnull().sum().sum()
        if null_count > 0:
            cleaned_df = cleaned_df.fillna(fill_value)
            print(f"Filled {null_count} null values with {fill_value}")
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing columns: {missing_cols}")
    
    if df.empty:
        validation_results['warnings'].append("DataFrame is empty")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        validation_results['warnings'].append("No numeric columns found")
    
    return validation_results

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    method (str): 'iqr' or 'zscore'
    threshold (float): Threshold for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data = df[column].dropna()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((df[column] - mean) / std)
        mask = z_scores <= threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    removed_count = len(df) - mask.sum()
    print(f"Removed {removed_count} outliers from column '{column}'")
    
    return df[mask].reset_index(drop=True)

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to normalize, None for all numeric columns
    method (str): 'minmax' or 'zscore'
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    normalized_df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'minmax':
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            if col_max != col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        
        elif method == 'zscore':
            col_mean = normalized_df[col].mean()
            col_std = normalized_df[col].std()
            if col_std != 0:
                normalized_df[col] = (normalized_df[col] - col_mean) / col_std
    
    return normalized_df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5, 5],
        'value': [10, 20, 20, None, 40, 50, 60],
        'category': ['A', 'B', 'B', 'C', 'D', 'E', 'F']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    validation = validate_data(cleaned, required_columns=['id', 'value'])
    print("Validation Results:")
    print(validation)