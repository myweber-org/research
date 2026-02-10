
import pandas as pd

def clean_dataframe(df):
    """
    Remove rows with any null values and standardize column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = (
        df_cleaned.columns
        .str.lower()
        .str.replace(' ', '_')
    )
    
    return df_cleanedimport pandas as pd
import numpy as np

def remove_missing_values(df, threshold=0.5):
    """
    Remove columns with missing values exceeding threshold percentage.
    """
    missing_percent = df.isnull().sum() / len(df)
    columns_to_drop = missing_percent[missing_percent > threshold].index
    return df.drop(columns=columns_to_drop)

def normalize_numeric_columns(df, columns=None):
    """
    Normalize specified numeric columns using min-max scaling.
    If columns is None, normalize all numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

def encode_categorical(df, columns=None, method='onehot'):
    """
    Encode categorical columns using specified method.
    Supported methods: 'onehot', 'label'
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns
    
    if method == 'onehot':
        return pd.get_dummies(df, columns=columns, drop_first=True)
    elif method == 'label':
        df_encoded = df.copy()
        for col in columns:
            if col in df.columns:
                df_encoded[col] = pd.Categorical(df[col]).codes
        return df_encoded
    else:
        raise ValueError("Method must be 'onehot' or 'label'")

def clean_dataset(df, missing_threshold=0.5, normalize=True, encode=True):
    """
    Complete data cleaning pipeline.
    """
    df_clean = df.copy()
    df_clean = remove_missing_values(df_clean, missing_threshold)
    
    if normalize:
        df_clean = normalize_numeric_columns(df_clean)
    
    if encode:
        df_clean = encode_categorical(df_clean)
    
    return df_clean
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', remove_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    missing_strategy (str): Strategy for handling missing values. 
                           Options: 'mean', 'median', 'mode', 'drop'
    remove_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif missing_strategy == 'median':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif missing_strategy == 'mode':
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
    elif missing_strategy == 'drop':
        df_clean = df_clean.dropna()
    
    # Remove duplicates
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        if removed > 0:
            print(f"Removed {removed} duplicate row(s)")
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
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
        'issues': [],
        'summary': {}
    }
    
    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['issues'].append('Input is not a pandas DataFrame')
        return validation_results
    
    # Check for empty DataFrame
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing required columns: {missing_cols}')
    
    # Calculate summary statistics
    validation_results['summary'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_results

def normalize_numeric_columns(df, columns=None, method='minmax'):
    """
    Normalize numeric columns in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to normalize. If None, normalize all numeric columns.
    method (str): Normalization method. Options: 'minmax', 'zscore'
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    # Determine which columns to normalize
    if columns is None:
        columns = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[col]):
            if method == 'minmax':
                col_min = df_normalized[col].min()
                col_max = df_normalized[col].max()
                if col_max > col_min:  # Avoid division by zero
                    df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
            elif method == 'zscore':
                col_mean = df_normalized[col].mean()
                col_std = df_normalized[col].std()
                if col_std > 0:  # Avoid division by zero
                    df_normalized[col] = (df_normalized[col] - col_mean) / col_std
    
    return df_normalized

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 5],
        'B': [10, 20, 30, np.nan, 50, 50],
        'C': ['x', 'y', 'z', 'x', 'y', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Validate data
    validation = validate_dataframe(df, required_columns=['A', 'B', 'C'])
    print("Validation Results:")
    print(validation)
    print("\n" + "="*50 + "\n")
    
    # Clean data
    df_clean = clean_dataset(df, missing_strategy='mean', remove_duplicates=True)
    print("Cleaned DataFrame:")
    print(df_clean)
    print("\n" + "="*50 + "\n")
    
    # Normalize numeric columns
    df_normalized = normalize_numeric_columns(df_clean, method='minmax')
    print("Normalized DataFrame:")
    print(df_normalized)