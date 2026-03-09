
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
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

def clean_dataset(df, numeric_columns):
    """
    Clean dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: Optional[str] = None,
    missing_strategy: str = "mean",
    drop_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and removing low-quality columns.
    
    Args:
        input_path: Path to input CSV file
        output_path: Optional path to save cleaned data
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        drop_threshold: Drop columns with missing ratio above this threshold
    
    Returns:
        Cleaned DataFrame
    """
    
    df = pd.read_csv(input_path)
    original_shape = df.shape
    
    # Remove columns with too many missing values
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    # Handle remaining missing values
    for column in df.columns:
        if df[column].isnull().any():
            if missing_strategy == "mean" and pd.api.types.is_numeric_dtype(df[column]):
                df[column].fillna(df[column].mean(), inplace=True)
            elif missing_strategy == "median" and pd.api.types.is_numeric_dtype(df[column]):
                df[column].fillna(df[column].median(), inplace=True)
            elif missing_strategy == "mode":
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif missing_strategy == "drop":
                df = df.dropna(subset=[column])
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Save cleaned data if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
    
    print(f"Data cleaning completed:")
    print(f"  Original shape: {original_shape}")
    print(f"  Cleaned shape: {df.shape}")
    print(f"  Columns removed: {len(columns_to_drop)}")
    print(f"  Rows removed: {original_shape[0] - df.shape[0]}")
    
    return df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame for common data quality issues.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": df.duplicated().sum(),
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": len(df.select_dtypes(include=['object']).columns),
        "date_columns": len(df.select_dtypes(include=['datetime']).columns)
    }
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    validation_results["constant_columns"] = constant_cols
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, np.nan, np.nan, 4, 5],
        'C': ['x', 'y', 'z', 'x', 'y'],
        'D': [10, 20, 30, 40, 50]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv("sample_data.csv", index=False)
    
    cleaned_df = clean_csv_data(
        "sample_data.csv",
        "cleaned_data.csv",
        missing_strategy="mean",
        drop_threshold=0.3
    )
    
    validation = validate_dataframe(cleaned_df)
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for identifying duplicates
        keep: determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by handling missing values.
    
    Args:
        df: pandas DataFrame
        column_name: name of the column to clean
        fill_method: method to fill missing values ('mean', 'median', 'zero')
    
    Returns:
        DataFrame with cleaned column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise ValueError(f"Column '{column_name}' is not numeric")
    
    missing_count = df[column_name].isna().sum()
    
    if missing_count > 0:
        if fill_method == 'mean':
            fill_value = df[column_name].mean()
        elif fill_method == 'median':
            fill_value = df[column_name].median()
        elif fill_method == 'zero':
            fill_value = 0
        else:
            raise ValueError("fill_method must be 'mean', 'median', or 'zero'")
        
        df[column_name] = df[column_name].fillna(fill_value)
        print(f"Filled {missing_count} missing value(s) in '{column_name}' with {fill_method}: {fill_value}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
    
    Returns:
        Boolean indicating if validation passed
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True