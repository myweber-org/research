import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: bool, whether to drop duplicate rows
        fill_method: str or value, method to fill missing values
                    ('mean', 'median', 'mode', or specific value)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method is not None:
        if fill_method == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_method == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_method == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        else:
            cleaned_df = cleaned_df.fillna(fill_method)
    else:
        cleaned_df = cleaned_df.dropna()
    
    # Remove duplicates
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
        min_rows: minimum number of rows required
    
    Returns:
        tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Dataset is valid"

def get_dataset_stats(df):
    """
    Get basic statistics about the dataset.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        dict with dataset statistics
    """
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': list(df.select_dtypes(include=['number']).columns),
        'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    return stats

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, None, 4, 5, 5],
        'B': ['x', 'y', 'z', None, 'x', 'x'],
        'C': [10.5, 20.3, 30.1, 40.7, None, 10.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nDataset statistics:")
    print(get_dataset_stats(df))
    
    # Clean the dataset
    cleaned = clean_dataset(df, drop_duplicates=True, fill_method='mean')
    print("\nCleaned dataset:")
    print(cleaned)
    
    # Validate the cleaned dataset
    is_valid, message = validate_dataset(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    
    return filtered_df

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
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original dataset shape:", df.shape)
    print("Original statistics for column 'A':")
    print(calculate_summary_statistics(df, 'A'))
    
    cleaned_df = clean_dataset(df, ['A'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Cleaned statistics for column 'A':")
    print(calculate_summary_statistics(cleaned_df, 'A'))
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, convert_types=True):
    """
    Clean a pandas DataFrame by removing duplicates, filling missing values,
    and converting data types where appropriate.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
                print(f"Filled missing values in numeric column '{column}' with median")
        
        for column in cleaned_df.select_dtypes(include=['object']).columns:
            if cleaned_df[column].isnull().any():
                cleaned_df[column] = cleaned_df[column].fillna('Unknown')
                print(f"Filled missing values in text column '{column}' with 'Unknown'")
    
    if convert_types:
        for column in cleaned_df.select_dtypes(include=['object']).columns:
            try:
                cleaned_df[column] = pd.to_datetime(cleaned_df[column], errors='ignore')
                if cleaned_df[column].dtype != 'object':
                    print(f"Converted column '{column}' to datetime")
            except:
                pass
        
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].apply(float.is_integer).all():
                cleaned_df[column] = cleaned_df[column].astype(int)
                print(f"Converted column '{column}' to integer")
    
    return cleaned_df

def validate_dataset(df, required_columns=None, unique_constraints=None):
    """
    Validate dataset structure and constraints.
    """
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    if unique_constraints:
        for constraint in unique_constraints:
            if df[constraint].duplicated().any():
                raise ValueError(f"Duplicate values found in unique constraint column: {constraint}")
    
    return True

def sample_data_processing():
    """
    Example usage of the data cleaning functions.
    """
    data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
        'age': [25, 30, 30, None, 35, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.0],
        'date': ['2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df)
    print("\nCleaned dataset:")
    print(cleaned)
    
    try:
        validate_dataset(cleaned, required_columns=['id', 'name'], unique_constraints=['id'])
        print("\nDataset validation passed")
    except ValueError as e:
        print(f"\nDataset validation failed: {e}")

if __name__ == "__main__":
    sample_data_processing()