
import pandas as pd
import numpy as np

def clean_dataset(df, drop_threshold=0.5, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_threshold (float): Threshold for dropping columns with too many nulls (0.0 to 1.0)
    fill_strategy (str): Strategy for filling remaining nulls ('mean', 'median', 'mode', or 'constant')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Standardize column names
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    # Drop columns with too many null values
    null_ratio = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = null_ratio[null_ratio > drop_threshold].index
    df_clean = df_clean.drop(columns=cols_to_drop)
    
    # Fill remaining null values based on strategy
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            if fill_strategy == 'mean' and np.issubdtype(df_clean[col].dtype, np.number):
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif fill_strategy == 'median' and np.issubdtype(df_clean[col].dtype, np.number):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif fill_strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0, inplace=True)
            elif fill_strategy == 'constant':
                df_clean[col].fillna(0, inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    return df_clean

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic quality checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_count': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['has_all_required_columns'] = len(missing_columns) == 0
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', None, 'David', 'Alice'],
        'Age': [25, 30, None, 35, 25],
        'Score': [85.5, 92.0, 78.5, None, 85.5],
        'Department': ['HR', 'IT', 'IT', 'Finance', 'HR']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataset(df))
    
    cleaned_df = clean_dataset(df, drop_threshold=0.3, fill_strategy='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Validation Results:")
    print(validate_dataset(cleaned_df))