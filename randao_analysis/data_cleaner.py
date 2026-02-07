import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (bool): Whether to fill missing values
    fill_value: Value to use for filling missing data
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    
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

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    method (str): 'iqr' for interquartile range or 'zscore' for standard deviations
    threshold (float): Threshold for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
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
    
    return df[mask]
import pandas as pd

def clean_dataset(df, id_column='id'):
    """
    Remove duplicate rows based on an ID column and standardize column names.
    """
    # Remove duplicates
    df_clean = df.drop_duplicates(subset=[id_column], keep='first')
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def validate_data(df, required_columns):
    """
    Check if required columns exist in the DataFrame.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'ID': [1, 2, 2, 3, 4],
        'Product Name': ['A', 'B', 'B', 'C', 'D'],
        'Price': [100, 200, 200, 300, 400]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, id_column='ID')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_data(cleaned_df, ['id', 'product_name', 'price'])
        print("\nData validation passed.")
    except ValueError as e:
        print(f"\nData validation failed: {e}")