
import pandas as pd
import numpy as np

def clean_dataset(df, duplicate_threshold=0.8, fill_strategy='median'):
    """
    Clean dataset by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    duplicate_threshold (float): Threshold for considering rows as duplicates
    fill_strategy (str): Strategy for filling missing values ('median', 'mean', 'mode')
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    
    original_shape = df.shape
    print(f"Original dataset shape: {original_shape}")
    
    # Remove exact duplicates
    df_cleaned = df.drop_duplicates()
    print(f"Removed {original_shape[0] - df_cleaned.shape[0]} exact duplicates")
    
    # Remove approximate duplicates based on threshold
    if duplicate_threshold < 1.0:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            similarity_matrix = df_cleaned[numeric_cols].corr().abs()
            high_corr_pairs = np.where(similarity_matrix > duplicate_threshold)
            high_corr_pairs = [(numeric_cols[i], numeric_cols[j]) 
                              for i, j in zip(*high_corr_pairs) 
                              if i < j]
            
            if high_corr_pairs:
                cols_to_drop = set()
                for col1, col2 in high_corr_pairs:
                    if col1 not in cols_to_drop and col2 not in cols_to_drop:
                        cols_to_drop.add(col2)
                
                df_cleaned = df_cleaned.drop(columns=list(cols_to_drop))
                print(f"Removed {len(cols_to_drop)} highly correlated columns")
    
    # Handle missing values
    missing_count = df_cleaned.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values")
        
        for column in df_cleaned.columns:
            if df_cleaned[column].isnull().any():
                if fill_strategy == 'median' and pd.api.types.is_numeric_dtype(df_cleaned[column]):
                    fill_value = df_cleaned[column].median()
                elif fill_strategy == 'mean' and pd.api.types.is_numeric_dtype(df_cleaned[column]):
                    fill_value = df_cleaned[column].mean()
                elif fill_strategy == 'mode':
                    fill_value = df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else None
                else:
                    fill_value = None
                
                if fill_value is not None:
                    df_cleaned[column] = df_cleaned[column].fillna(fill_value)
                    print(f"Filled missing values in '{column}' with {fill_strategy}: {fill_value}")
                else:
                    # Drop rows with missing values if no valid fill strategy
                    df_cleaned = df_cleaned.dropna(subset=[column])
        
        remaining_missing = df_cleaned.isnull().sum().sum()
        print(f"Remaining missing values after cleaning: {remaining_missing}")
    
    # Remove constant columns
    constant_cols = [col for col in df_cleaned.columns if df_cleaned[col].nunique() <= 1]
    if constant_cols:
        df_cleaned = df_cleaned.drop(columns=constant_cols)
        print(f"Removed {len(constant_cols)} constant columns: {constant_cols}")
    
    final_shape = df_cleaned.shape
    print(f"Final dataset shape: {final_shape}")
    print(f"Removed {original_shape[0] - final_shape[0]} rows and {original_shape[1] - final_shape[1]} columns")
    
    return df_cleaned

def validate_dataframe(df):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_dataframe': isinstance(df, pd.DataFrame),
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'has_duplicates': df.duplicated().any(),
        'missing_values': df.isnull().sum().sum(),
        'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 4, 5, None],
        'B': [1.1, 2.2, 2.2, 4.4, 5.5, 6.6],
        'C': [1, 1, 1, 1, 1, 1],  # Constant column
        'D': ['a', 'b', 'b', 'c', 'd', 'e']
    }
    
    df = pd.DataFrame(sample_data)
    print("Initial validation:")
    print(validate_dataframe(df))
    
    cleaned_df = clean_dataset(df, duplicate_threshold=0.9, fill_strategy='median')
    
    print("\nFinal validation:")
    print(validate_dataframe(cleaned_df))
import pandas as pd

def clean_dataset(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values.
    rename_columns (bool): If True, rename columns to lowercase with underscores.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df.columns = [col.strip().lower().replace(' ', '_') for col in cleaned_df.columns]
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'Name': ['Alice', 'Bob', None, 'Charlie'],
        'Age': [25, 30, 35, None],
        'City': ['New York', 'London', 'Paris', 'Tokyo']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df)
    print(cleaned)
    print(f"\nDataFrame validation: {validate_dataframe(cleaned)}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    cleaned_data = remove_outliers_iqr(data, column)
    
    stats = {
        'mean': np.mean(cleaned_data[column]),
        'median': np.median(cleaned_data[column]),
        'std': np.std(cleaned_data[column])
    }
    return statsimport pandas as pd
import numpy as np
from typing import Optional, List

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in specified columns using specified strategy.
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            
            df_filled[col] = df[col].fillna(fill_value)
        else:
            df_filled[col] = df[col].fillna('Unknown')
    
    return df_filled

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a numeric column to range [0, 1].
    """
    df_normalized = df.copy()
    if df[column].dtype in [np.float64, np.int64]:
        col_min = df[column].min()
        col_max = df[column].max()
        
        if col_max != col_min:
            df_normalized[column] = (df[column] - col_min) / (col_max - col_min)
    
    return df_normalized

def clean_dataframe(df: pd.DataFrame, 
                   remove_dups: bool = True,
                   fill_na: bool = True,
                   normalize_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Basic validation of DataFrame structure and content.
    """
    if df.empty:
        return False
    
    if df.isnull().all().any():
        return False
    
    return True