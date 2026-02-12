
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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
    
    Args:
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
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean. 
                                 If None, cleans all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10, 30, 31],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 200, 55],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nOriginal shape:", df.shape)
    
    cleaned_df = clean_numeric_data(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned shape:", cleaned_df.shape)
    
    for column in df.columns:
        stats = calculate_basic_stats(df, column)
        print(f"\nStatistics for '{column}':")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
import pandas as pd
import sys

def remove_duplicates(input_file, output_file, column_name):
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        df.drop_duplicates(subset=[column_name], keep='first', inplace=True)
        final_count = len(df)
        df.to_csv(output_file, index=False)
        return initial_count - final_count
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except KeyError:
        print(f"Error: Column '{column_name}' not found in the CSV.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python data_cleaner.py <input_file> <output_file> <column_name>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    column_name = sys.argv[3]
    
    removed = remove_duplicates(input_file, output_file, column_name)
    print(f"Removed {removed} duplicate entries based on column '{column_name}'.")
import pandas as pd

def clean_dataset(df):
    """
    Remove null values and duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def filter_by_column(df, column_name, threshold):
    """
    Filter DataFrame rows where column value is greater than threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to filter by.
        threshold (float): Threshold value.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    filtered_df = df[df[column_name] > threshold]
    return filtered_df

def calculate_statistics(df):
    """
    Calculate basic statistics for numerical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        dict: Dictionary containing statistics.
    """
    stats = {}
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numerical_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return stats