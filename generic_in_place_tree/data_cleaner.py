
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
        columns (list): List of columns to process, None for all columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns
    
    for col in columns:
        if df_clean[col].isnull().any():
            if strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
    
    return df_clean

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to process, None for all numeric columns
        threshold (float): IQR multiplier threshold
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if df_clean[col].dtype in [np.float64, np.int64]:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_data(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to standardize, None for all numeric columns
    
    Returns:
        pd.DataFrame: Standardized DataFrame
    """
    df_std = df.copy()
    
    if columns is None:
        columns = df_std.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if df_std[col].dtype in [np.float64, np.int64]:
            mean_val = df_std[col].mean()
            std_val = df_std[col].std()
            if std_val > 0:
                df_std[col] = (df_std[col] - mean_val) / std_val
    
    return df_std

def clean_dataset(df, missing_strategy='mean', outlier_threshold=1.5, standardize=True):
    """
    Complete data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values
        outlier_threshold (float): Threshold for outlier removal
        standardize (bool): Whether to standardize numeric columns
    
    Returns:
        pd.DataFrame: Cleaned and processed DataFrame
    """
    df_clean = clean_missing_values(df, strategy=missing_strategy)
    df_clean = remove_outliers_iqr(df_clean, threshold=outlier_threshold)
    
    if standardize:
        df_clean = standardize_data(df_clean)
    
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, np.nan, 4, 5, 100],
        'feature2': [10, 20, 30, 40, 50, 60],
        'feature3': [100, 200, 300, 400, 500, 600]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df)
    print(cleaned_df)
import pandas as pd

def clean_dataset(df, subset=None, fill_method='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Columns to consider for duplicate removal.
        fill_method (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates(subset=subset, keep='first')
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
    categorical_cols = cleaned_df.select_dtypes(exclude=['number']).columns
    
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'mean':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
    elif fill_method == 'median':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
    elif fill_method == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    # Fill remaining categorical columns with 'Unknown'
    cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna('Unknown')
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    return True, "Data validation passed"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'id': [1, 2, 2, 3, 4],
#         'value': [10.5, None, 15.0, 20.0, None],
#         'category': ['A', 'B', None, 'A', 'C']
#     }
#     df = pd.DataFrame(sample_data)
#     cleaned = clean_dataset(df, subset=['id'], fill_method='mean')
#     print("Cleaned DataFrame:")
#     print(cleaned)
#     
#     is_valid, message = validate_data(cleaned, required_columns=['id', 'value'], min_rows=2)
#     print(f"Validation: {is_valid}, Message: {message}")