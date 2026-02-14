
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, processes all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'pressure': [1013, 1012, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 950]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nOriginal statistics for 'temperature':")
    print(calculate_basic_stats(df, 'temperature'))
    
    cleaned_df = clean_numeric_data(df, ['temperature', 'pressure'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned statistics for 'temperature':")
    print(calculate_basic_stats(cleaned_df, 'temperature'))
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, missing_strategy='drop'):
    """
    Clean CSV data by handling missing values and standardizing columns.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str, optional): Path for cleaned output CSV. If None, adds '_cleaned' suffix
    missing_strategy (str): Strategy for handling missing values: 'drop', 'mean', 'median', or 'zero'
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    # Read input data
    df = pd.read_csv(input_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    # Handle missing values based on strategy
    if missing_strategy == 'drop':
        df_cleaned = df.dropna()
    elif missing_strategy == 'mean':
        df_cleaned = df.fillna(df.mean(numeric_only=True))
    elif missing_strategy == 'median':
        df_cleaned = df.fillna(df.median(numeric_only=True))
    elif missing_strategy == 'zero':
        df_cleaned = df.fillna(0)
    else:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    # Standardize column names: lowercase, replace spaces with underscores
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')
    
    # Remove duplicate rows
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    duplicates_removed = initial_rows - len(df_cleaned)
    
    # Determine output path
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = input_path_obj.parent / f"{input_path_obj.stem}_cleaned.csv"
    
    # Save cleaned data
    df_cleaned.to_csv(output_path, index=False)
    
    print(f"Cleaned data shape: {df_cleaned.shape}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Cleaned data saved to: {output_path}")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate
    required_columns (list, optional): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {}
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing required columns: {missing_columns}')
    
    # Add summary statistics
    validation_results['summary'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'missing_values': df.isnull().sum().sum()
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
        'Age': [25, 30, None, 35, 40],
        'Salary': [50000, 60000, 70000, None, 90000],
        'Department': ['HR', 'IT', 'IT', 'HR', 'IT']
    }
    
    # Create sample CSV
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data('sample_data.csv', missing_strategy='mean')
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned_df, required_columns=['name', 'age', 'salary'])
    print(f"\nValidation results: {validation}")