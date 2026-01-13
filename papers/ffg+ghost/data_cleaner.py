
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_check (list): List of column names to check for duplicates
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        if columns_to_check:
            df_clean = df_clean.drop_duplicates(subset=columns_to_check)
        else:
            df_clean = df_clean.drop_duplicates()
    
    if fill_missing == 'drop':
        df_clean = df_clean.dropna()
    else:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = df_clean[col].mean()
                elif fill_missing == 'median':
                    fill_value = df_clean[col].median()
                elif fill_missing == 'mode':
                    fill_value = df_clean[col].mode()[0]
                else:
                    fill_value = 0
                
                df_clean[col] = df_clean[col].fillna(fill_value)
    
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna('Unknown')
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
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

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    method (str): Method to detect outliers ('iqr' or 'zscore')
    threshold (float): Threshold for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        return df
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        z_scores = np.abs((df[column] - mean) / std)
        mask = z_scores <= threshold
    
    else:
        return df
    
    return df[mask]

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
        'age': [25, 30, None, 40, 35, 35, 150],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'], fill_missing='mean')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    is_valid, message = validate_data(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    print(f"\nValidation: {message}")
    
    df_no_outliers = remove_outliers(cleaned_df, 'age', method='iqr')
    print(f"\nRows after removing age outliers: {len(df_no_outliers)}")