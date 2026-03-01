
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
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
    
    return filtered_df

def standardize_column(df, column):
    """
    Standardize a column to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to standardize.
    
    Returns:
    pd.DataFrame: DataFrame with standardized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = df[column].mean()
    std_val = df[column].std()
    
    if std_val == 0:
        return df
    
    df_copy = df.copy()
    df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    strategy (str): Imputation strategy ('mean', 'median', or 'drop').
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df.dropna(subset=numeric_cols)
    elif strategy == 'mean':
        impute_values = df[numeric_cols].mean()
    elif strategy == 'median':
        impute_values = df[numeric_cols].median()
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    df_filled = df.copy()
    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(impute_values)
    
    return df_filled

def clean_dataset(df, numeric_columns=None, outlier_removal=True, standardization=True, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_columns (list): List of numeric columns to process. If None, all numeric columns are used.
    outlier_removal (bool): Whether to remove outliers.
    standardization (bool): Whether to standardize columns.
    missing_strategy (str): Strategy for handling missing values.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if outlier_removal:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    if standardization:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = standardize_column(cleaned_df, col)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [5, np.nan, 15, 20, 25, 30, 35, 40, 45, 50]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataset(df, outlier_removal=True, standardization=True)
    print("Cleaned DataFrame:")
    print(cleaned)
import pandas as pd
import numpy as np

def clean_data(input_file, output_file):
    df = pd.read_csv(input_file)
    
    df = df.dropna()
    
    df = df.drop_duplicates()
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    print(f"Original rows: {len(pd.read_csv(input_file))}, Cleaned rows: {len(df)}")

if __name__ == "__main__":
    clean_data("raw_data.csv", "cleaned_data.csv")import csv
import sys

def clean_csv(input_file, output_file, key_column):
    """
    Remove duplicate rows based on a key column and convert numeric columns.
    """
    seen = set()
    cleaned_rows = []
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            
            for row in reader:
                key = row.get(key_column)
                if key is None:
                    continue
                
                if key not in seen:
                    seen.add(key)
                    processed_row = {}
                    for field in fieldnames:
                        value = row[field]
                        if value.replace('.', '', 1).isdigit():
                            if '.' in value:
                                processed_row[field] = float(value)
                            else:
                                processed_row[field] = int(value)
                        else:
                            processed_row[field] = value
                    cleaned_rows.append(processed_row)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned_rows)
            
        print(f"Cleaned data saved to {output_file}")
        print(f"Removed {len(seen) - len(cleaned_rows)} duplicate rows")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python data_cleaner.py <input.csv> <output.csv> <key_column>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key_column = sys.argv[3]
    
    clean_csv(input_file, output_file, key_column)
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to analyze
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        Boolean mask of outliers
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to analyze
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Normalized Series
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    return (data[column] - min_val) / (max_val - min_val)

def standardize_data(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Standardized Series
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize=False):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names to clean
        outlier_method: 'iqr' or 'zscore' (default 'iqr')
        normalize: whether to normalize data (default False)
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        # Handle missing values
        cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].median())
        
        # Remove outliers
        if outlier_method == 'iqr':
            outliers = detect_outliers_iqr(cleaned_data, column)
            cleaned_data = cleaned_data[~outliers]
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
    
    # Normalize if requested
    if normalize:
        for column in numeric_columns:
            if column in cleaned_data.columns:
                cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
                cleaned_data[f'{column}_standardized'] = standardize_data(cleaned_data, column)
    
    return cleaned_data.reset_index(drop=True)

def get_summary_statistics(data, numeric_columns):
    """
    Generate summary statistics for numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names
    
    Returns:
        DataFrame with summary statistics
    """
    summary = pd.DataFrame()
    
    for column in numeric_columns:
        if column in data.columns:
            col_data = data[column].dropna()
            if len(col_data) > 0:
                stats_dict = {
                    'column': column,
                    'count': len(col_data),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    '25%': col_data.quantile(0.25),
                    'median': col_data.median(),
                    '75%': col_data.quantile(0.75),
                    'max': col_data.max(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis()
                }
                summary = pd.concat([summary, pd.DataFrame([stats_dict])], ignore_index=True)
    
    return summary

# Example usage demonstration
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000)
    })
    
    # Add some outliers
    sample_data.loc[10, 'feature_a'] = 500
    sample_data.loc[20, 'feature_b'] = 1000
    
    # Clean the data
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    cleaned = clean_dataset(sample_data, numeric_cols, outlier_method='iqr', normalize=True)
    
    # Get summary statistics
    original_stats = get_summary_statistics(sample_data, numeric_cols)
    cleaned_stats = get_summary_statistics(cleaned, numeric_cols)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Cleaned data shape: {cleaned.shape}")
    print(f"Rows removed: {len(sample_data) - len(cleaned)}")
    
    print("\nOriginal statistics:")
    print(original_stats[['column', 'mean', 'std', 'min', 'max']].head())
    
    print("\nCleaned statistics:")
    print(cleaned_stats[['column', 'mean', 'std', 'min', 'max']].head())