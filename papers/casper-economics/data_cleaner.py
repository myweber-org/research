import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Strategy for missing value imputation ('mean', 'median', 'mode')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    
    df_clean = df.copy()
    
    # Handle missing values
    for column in df_clean.columns:
        if df_clean[column].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df_clean[column].mean()
            elif strategy == 'median':
                fill_value = df_clean[column].median()
            elif strategy == 'mode':
                fill_value = df_clean[column].mode()[0]
            else:
                fill_value = 0
                
            df_clean[column].fillna(fill_value, inplace=True)
        else:
            df_clean[column].fillna(df_clean[column].mode()[0], inplace=True)
    
    # Remove outliers using Z-score method
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((df_clean[numeric_cols] - df_clean[numeric_cols].mean()) / df_clean[numeric_cols].std())
    
    outlier_mask = (z_scores < outlier_threshold).all(axis=1)
    df_clean = df_clean[outlier_mask].reset_index(drop=True)
    
    return df_clean

def validate_data(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {},
        'data_types': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['is_valid'] = False
            validation_results['missing_columns'] = missing
    
    for column in df.columns:
        null_count = df[column].isnull().sum()
        validation_results['null_counts'][column] = null_count
        validation_results['data_types'][column] = str(df[column].dtype)
        
        if null_count > 0:
            validation_results['is_valid'] = False
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['a', 'b', 'c', 'd', 'e']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_data(df))
    
    cleaned_df = clean_dataset(df, strategy='median', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned_df)