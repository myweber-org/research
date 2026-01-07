import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to check for missing values.
                                 If None, checks all columns.
    
    Returns:
        pd.DataFrame: DataFrame with rows containing missing values removed
    """
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def fill_missing_with_mean(df, columns=None):
    """
    Fill missing values with column mean.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to fill.
                                 If None, fills all numeric columns.
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df_filled[col] = df[col].fillna(df[col].mean())
    
    return df_filled

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to check for outliers.
                                 If None, checks all numeric columns.
        threshold (float): Multiplier for IQR (default: 1.5)
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to standardize.
                                 If None, standardizes all numeric columns.
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    df_standardized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df_standardized[col] = (df[col] - mean) / std
    
    return df_standardized

def clean_dataset(df, missing_strategy='remove', outlier_removal=True, standardization=False):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values.
                               Options: 'remove', 'mean', 'ignore'
        outlier_removal (bool): Whether to remove outliers using IQR method
        standardization (bool): Whether to standardize numeric columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df)
    elif missing_strategy == 'mean':
        cleaned_df = fill_missing_with_mean(cleaned_df)
    
    # Remove outliers
    if outlier_removal:
        cleaned_df = remove_outliers_iqr(cleaned_df)
    
    # Standardize columns
    if standardization:
        cleaned_df = standardize_columns(cleaned_df)
    
    return cleaned_df
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = df[column].fillna(df[column].median())
    
    # Remove outliers using z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    for column in numeric_cols:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = load_and_clean_data(input_file)
        save_cleaned_data(cleaned_df, output_file)
        print(f"Original shape: {pd.read_csv(input_file).shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")
    except FileNotFoundError:
        print(f"Error: {input_file} not found")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
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
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
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

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10, 'value'] = 500
    df.loc[20, 'value'] = -100
    
    print("Original DataFrame shape:", df.shape)
    print("Original summary statistics:")
    print(calculate_summary_statistics(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary statistics:")
    print(calculate_summary_statistics(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    result_df = example_usage()
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    
    # Handle missing values
    if fill_missing:
        if columns_to_check is None:
            columns_to_check = df_cleaned.columns
        
        for column in columns_to_check:
            if df_cleaned[column].dtype in [np.float64, np.int64]:
                # Fill numeric columns with median
                df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())
            elif df_cleaned[column].dtype == 'object':
                # Fill categorical columns with mode
                df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else 'Unknown')
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    # Print cleaning summary
    print(f"Original dataset shape: {original_shape}")
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Missing values filled: {fill_missing}")
    
    return df_cleaned

def validate_data(df, required_columns=None):
    """
    Validate that the DataFrame meets basic data quality requirements.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    # Create sample data with some issues
    data = {
        'id': [1, 2, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve', 'Eve'],
        'age': [25, 30, 30, None, 35, 40, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5, 95.5]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, fill_missing=True)
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    try:
        validate_data(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
        print("\nData validation passed!")
    except ValueError as e:
        print(f"\nData validation failed: {e}")