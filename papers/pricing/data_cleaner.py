
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
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'q1': df[column].quantile(0.25),
        'q3': df[column].quantile(0.75)
    }
    
    return stats

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of summary statistics for each cleaned column
    """
    cleaned_df = df.copy()
    all_stats = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            all_stats[column] = stats
    
    return cleaned_df, all_stats

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample data with outliers
    data = {
        'temperature': np.concatenate([
            np.random.normal(20, 2, 90),
            np.array([40, 45, 50, -10, -5])
        ]),
        'humidity': np.concatenate([
            np.random.normal(50, 5, 90),
            np.array([90, 95, 100, 0, 5])
        ]),
        'pressure': np.random.normal(1013, 10, 95)
    }
    
    sample_df = pd.DataFrame(data)
    
    print("Original dataset shape:", sample_df.shape)
    print("\nOriginal summary statistics:")
    print(sample_df.describe())
    
    # Clean the dataset
    columns_to_clean = ['temperature', 'humidity']
    cleaned_df, stats = clean_dataset(sample_df, columns_to_clean)
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaning statistics:")
    for column, column_stats in stats.items():
        print(f"\n{column}:")
        for stat_name, stat_value in column_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")
import pandas as pd
import numpy as np
from pathlib import Path

def load_csv_data(file_path):
    """
    Load CSV file into pandas DataFrame.
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        DataFrame containing loaded data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        print(f"Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: Specific columns to clean, None for all columns
    
    Returns:
        Cleaned DataFrame
    """
    if df is None or df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df.columns:
            if df[col].isnull().any():
                null_count = df[col].isnull().sum()
                print(f"Column '{col}' has {null_count} missing values")
                
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                    df_clean[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                    df_clean[col].fillna(df[col].median(), inplace=True)
                elif strategy == 'mode':
                    df_clean[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
                elif strategy == 'drop':
                    df_clean = df_clean.dropna(subset=[col])
                else:
                    df_clean[col].fillna(0, inplace=True)
    
    return df_clean

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    if df is None or df.empty:
        return df
    
    initial_rows = len(df)
    df_deduped = df.drop_duplicates(subset=subset, keep='first')
    removed_count = initial_rows - len(df_deduped)
    
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return df_deduped

def normalize_numeric_columns(df, columns=None):
    """
    Normalize numeric columns to 0-1 range.
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize, None for all numeric columns
    
    Returns:
        DataFrame with normalized columns
    """
    if df is None or df.empty:
        return df
    
    df_normalized = df.copy()
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max > col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
                print(f"Normalized column '{col}'")
    
    return df_normalized

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
    
    Returns:
        Boolean indicating success
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Successfully saved cleaned data to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

def process_data_pipeline(input_file, output_file, clean_strategy='mean'):
    """
    Complete data cleaning pipeline.
    
    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path
        clean_strategy: Strategy for handling missing values
    
    Returns:
        Boolean indicating pipeline success
    """
    print(f"Starting data cleaning pipeline for {input_file}")
    
    df = load_csv_data(input_file)
    if df is None:
        return False
    
    df = clean_missing_values(df, strategy=clean_strategy)
    df = remove_duplicates(df)
    df = normalize_numeric_columns(df)
    
    success = save_cleaned_data(df, output_file)
    
    if success:
        print(f"Pipeline completed successfully. Output: {output_file}")
    else:
        print("Pipeline failed")
    
    return success

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'value': [10.5, 15.2, None, 18.7, 12.3, 15.2, None, 20.1, 22.5, 19.8],
        'category': ['A', 'B', 'A', 'C', 'B', 'B', 'A', 'C', 'A', 'B'],
        'score': [85, 92, 78, None, 88, 92, 76, 95, 89, 91]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_input = "test_data.csv"
    test_output = "cleaned_test_data.csv"
    
    test_df.to_csv(test_input, index=False)
    
    process_data_pipeline(test_input, test_output)
    
    Path(test_input).unlink(missing_ok=True)
    Path(test_output).unlink(missing_ok=True)