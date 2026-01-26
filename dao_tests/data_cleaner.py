
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if subset is None:
        subset = dataframe.columns.tolist()
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    df_copy = dataframe.copy()
    
    for column in columns:
        if column in df_copy.columns:
            df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
    
    return df_copy

def validate_dataframe(dataframe, required_columns):
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if all required columns are present
    """
    existing_columns = set(dataframe.columns)
    required_set = set(required_columns)
    
    return required_set.issubset(existing_columns)import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    """
    Load a CSV file and perform data cleaning operations.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Remove duplicate rows
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - df.shape[0]
    print(f"Removed {duplicates_removed} duplicate rows.")

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"Filled missing values in {col} with median.")

    # Remove outliers using Z-score for numeric columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    outlier_mask = (z_scores < 3).all(axis=1)
    df_clean = df[outlier_mask]
    outliers_removed = df.shape[0] - df_clean.shape[0]
    print(f"Removed {outliers_removed} outliers based on Z-score.")

    # Normalize numeric columns to range [0, 1]
    for col in numeric_cols:
        min_val = df_clean[col].min()
        max_val = df_clean[col].max()
        if max_val > min_val:
            df_clean[col] = (df_clean[col] - min_val) / (max_val - min_val)
            print(f"Normalized column {col}.")

    print(f"Final cleaned data shape: {df_clean.shape}")
    return df_clean

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)