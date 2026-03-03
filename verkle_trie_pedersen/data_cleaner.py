
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (bool): Whether to fill missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
            elif cleaned_df[column].dtype == 'object':
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
        print("Missing values have been filled")
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate the dataset for required columns and data integrity.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.isnull().sum().sum() > 0:
        print("Dataset contains missing values")
        return False
    
    return True

def main():
    sample_data = {
        'id': [1, 2, 3, 4, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', None, 'David', 'Eve'],
        'age': [25, 30, None, 35, 35, 28],
        'score': [85.5, 92.0, 78.5, 88.0, 88.0, 91.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    is_valid = validate_dataset(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    print(f"\nDataset validation: {'PASS' if is_valid else 'FAIL'}")

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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

def process_dataset(file_path, column_name):
    """
    Load a dataset and clean it by removing outliers.
    
    Parameters:
    file_path (str): Path to CSV file
    column_name (str): Column to clean
    
    Returns:
    tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_stats = calculate_summary_statistics(df, column_name)
    cleaned_df = remove_outliers_iqr(df, column_name)
    cleaned_stats = calculate_summary_statistics(cleaned_df, column_name)
    
    return cleaned_df, original_stats, cleaned_stats

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'values': np.concatenate([
            np.random.normal(100, 10, 90),
            np.array([500, 600, 700])  # Outliers
        ])
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data shape:", df.shape)
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("Cleaned data shape:", cleaned_df.shape)
    
    stats = calculate_summary_statistics(cleaned_df, 'values')
    print("Cleaned statistics:", stats)
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

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    return stats

def clean_dataset(df, numeric_columns):
    """
    Clean a dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of removal statistics for each column
    """
    original_shape = df.shape
    removal_stats = {}
    
    for column in numeric_columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = df.shape[0]
            df = remove_outliers_iqr(df, column)
            removed_count = original_count - df.shape[0]
            removal_stats[column] = {
                'removed': removed_count,
                'percentage': (removed_count / original_count) * 100
            }
    
    final_shape = df.shape
    removal_stats['total_removed'] = original_shape[0] - final_shape[0]
    removal_stats['original_shape'] = original_shape
    removal_stats['final_shape'] = final_shape
    
    return df, removal_stats

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(300, 50, 10)
        ]),
        'score': np.concatenate([
            np.random.normal(50, 5, 95),
            np.random.normal(200, 30, 5)
        ])
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset shape:", df.shape)
    print("\nOriginal statistics for 'value':")
    print(calculate_statistics(df, 'value'))
    
    cleaned_df, stats = clean_dataset(df, ['value', 'score'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nRemoval statistics:")
    for col, col_stats in stats.items():
        if col not in ['total_removed', 'original_shape', 'final_shape']:
            print(f"{col}: {col_stats['removed']} outliers removed ({col_stats['percentage']:.2f}%)")
    
    print(f"\nTotal rows removed: {stats['total_removed']}")
    print(f"Original shape: {stats['original_shape']}")
    print(f"Final shape: {stats['final_shape']}")