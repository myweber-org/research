
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str or dict): Method to fill missing values.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, str):
            if fill_missing == 'mean':
                cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
            elif fill_missing == 'median':
                cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
            elif fill_missing == 'mode':
                cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
            elif fill_missing == 'ffill':
                cleaned_df = cleaned_df.fillna(method='ffill')
            elif fill_missing == 'bfill':
                cleaned_df = cleaned_df.fillna(method='bfill')
        elif isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    method (str): Outlier detection method ('iqr' or 'zscore').
    threshold (float): Threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    data = df[column].copy()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        z_scores = (data - mean) / std
        mask = abs(z_scores) <= threshold
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str): Path for cleaned CSV file (optional)
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    original_rows = len(df)
    print(f"Original data: {original_rows} rows, {len(df.columns)} columns")
    
    df = df.drop_duplicates()
    print(f"After removing duplicates: {len(df)} rows")
    
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        print(f"Missing values found:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count} missing values")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                if df[col].isnull().any():
                    fill_value = df[col].mean()
                    df[col] = df[col].fillna(fill_value)
                    print(f"Filled missing values in '{col}' with mean: {fill_value:.2f}")
        
        elif missing_strategy == 'median':
            for col in numeric_cols:
                if df[col].isnull().any():
                    fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    print(f"Filled missing values in '{col}' with median: {fill_value:.2f}")
        
        elif missing_strategy == 'drop':
            df = df.dropna()
            print(f"Removed rows with missing values: {len(df)} rows remaining")
        
        else:
            raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    print(f"Final data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Rows removed: {original_rows - len(df)}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {}
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append("DataFrame is empty")
    
    validation_results['summary']['rows'] = len(df)
    validation_results['summary']['columns'] = len(df.columns)
    validation_results['summary']['memory_usage'] = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        validation_results['summary']['numeric_stats'] = {}
        for col in numeric_cols:
            validation_results['summary']['numeric_stats'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'value': [10.5, 20.3, np.nan, 15.7, 20.3, np.nan, 25.1, 30.0],
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
        'score': [85, 92, 78, np.nan, 88, 91, 95, 87]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    print("Testing data cleaning utility...")
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv', 'mean')
    
    print("\nValidating cleaned data...")
    validation = validate_dataframe(cleaned_df, ['id', 'value', 'category', 'score'])
    
    print(f"Validation passed: {validation['is_valid']}")
    print(f"Data summary: {validation['summary']}")
    
    import os
    os.remove('test_data.csv')
    if os.path.exists('cleaned_data.csv'):
        os.remove('cleaned_data.csv')