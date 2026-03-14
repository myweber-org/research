
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'original_count': len(df),
        'cleaned_count': len(remove_outliers_iqr(df, column)),
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }
    
    return stats

def example_usage():
    """
    Example demonstrating the usage of data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.random.normal(300, 50, 5)
        ])
    }
    
    df = pd.DataFrame(data)
    print(f"Original data shape: {df.shape}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print(f"Cleaned data shape: {cleaned_df.shape}")
    
    stats = calculate_summary_statistics(df, 'values')
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    example_usage()import pandas as pd

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    column_mapping (dict): Optional dictionary to rename columns
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (str): Strategy to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Rename columns if mapping provided
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate the structure and content of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    numeric_columns (list): List of columns that should be numeric
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'non_numeric_columns': [],
        'null_counts': {}
    }
    
    # Check required columns
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    # Check numeric columns
    if numeric_columns:
        non_numeric = []
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric.append(col)
        if non_numeric:
            validation_results['non_numeric_columns'] = non_numeric
            validation_results['is_valid'] = False
    
    # Count null values
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_results['null_counts'][col] = null_count
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve'],
        'age': [25, 30, None, 40, 35, 35],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Validate the cleaned data
    validation = validate_data(
        cleaned, 
        required_columns=['id', 'name', 'age', 'score'],
        numeric_columns=['age', 'score']
    )
    print("Validation Results:")
    print(validation)