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
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        df: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    """
    Normalize column values to range [0, 1] using min-max scaling.
    
    Args:
        df: pandas DataFrame
        column: Column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        df[column + '_normalized'] = 0.5
    else:
        df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    
    return df

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric columns to process
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    
    return cleaned_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        df: pandas DataFrame
        column: Column name
    
    Returns:
        Dictionary with statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 21),
        'value': [10, 12, 15, 18, 20, 22, 25, 28, 30, 32,
                  35, 38, 40, 42, 45, 48, 50, 100, 120, 150]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    
    cleaned = clean_dataset(df, ['value'])
    print("\nCleaned data:")
    print(cleaned)
    
    stats = calculate_statistics(df, 'value')
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")