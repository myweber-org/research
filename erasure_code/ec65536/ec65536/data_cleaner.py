import pandas as pd
import numpy as np

def clean_dataset(df, drop_threshold=0.5, fill_strategy='median'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_threshold (float): Threshold for dropping columns with too many nulls (0.0 to 1.0)
    fill_strategy (str): Strategy for filling missing values ('median', 'mean', 'mode', or 'constant')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Standardize column names
    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Drop columns with too many null values
    null_ratio = df_clean.isnull().sum() / len(df_clean)
    columns_to_drop = null_ratio[null_ratio > drop_threshold].index
    df_clean = df_clean.drop(columns=columns_to_drop)
    
    # Fill missing values based on strategy
    for column in df_clean.columns:
        if df_clean[column].isnull().any():
            if fill_strategy == 'median' and pd.api.types.is_numeric_dtype(df_clean[column]):
                df_clean[column] = df_clean[column].fillna(df_clean[column].median())
            elif fill_strategy == 'mean' and pd.api.types.is_numeric_dtype(df_clean[column]):
                df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
            elif fill_strategy == 'mode':
                df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
            elif fill_strategy == 'constant':
                if pd.api.types.is_numeric_dtype(df_clean[column]):
                    df_clean[column] = df_clean[column].fillna(0)
                else:
                    df_clean[column] = df_clean[column].fillna('unknown')
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for common data quality issues.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    
    Returns:
    dict: Dictionary containing validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_values': int(df.isnull().sum().sum()),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': {col: str(df[col].dtype) for col in df.columns}
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
    
    return validation_results

# Example usage (commented out for production)
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = {
        'Customer ID': [1, 2, 3, 4, 5],
        'First Name': ['Alice', 'Bob', None, 'David', 'Eve'],
        'Last Name': ['Smith', 'Johnson', 'Williams', None, 'Brown'],
        'Age': [25, 30, 35, None, 28],
        'Salary': [50000, 60000, None, 75000, 55000],
        'Department': ['Sales', 'IT', 'IT', None, 'HR']
    }
    
    df_sample = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df_sample)
    print("\nValidation Results:")
    print(validate_dataset(df_sample))
    
    df_cleaned = clean_dataset(df_sample, drop_threshold=0.3, fill_strategy='median')
    print("\nCleaned DataFrame:")
    print(df_cleaned)
    print("\nCleaned Validation Results:")
    print(validate_dataset(df_cleaned))