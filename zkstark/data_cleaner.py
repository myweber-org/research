import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to remove duplicate rows. Default is True.
        fill_missing (str or dict): Strategy to fill missing values. 
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    
    return cleaned_df

def remove_outliers_iqr(df, columns, multiplier=1.5):
    """
    Remove outliers from specified columns using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to process.
        multiplier (float): IQR multiplier for outlier detection. Default is 1.5.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_columns(df, columns):
    """
    Standardize specified columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to standardize.
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns.
    """
    df_standardized = df.copy()
    
    for col in columns:
        if col in df_standardized.columns and pd.api.types.is_numeric_dtype(df_standardized[col]):
            mean = df_standardized[col].mean()
            std = df_standardized[col].std()
            if std > 0:
                df_standardized[col] = (df_standardized[col] - mean) / std
    
    return df_standardized

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, 4, 5, 100],
        'B': [10, 20, 20, 30, None, 50, 60],
        'C': [100, 200, 200, 300, 400, 500, 600]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    no_outliers = remove_outliers_iqr(cleaned, ['A', 'C'])
    print("\nDataFrame without outliers:")
    print(no_outliers)
    
    standardized = standardize_columns(no_outliers, ['A', 'C'])
    print("\nStandardized DataFrame:")
    print(standardized)