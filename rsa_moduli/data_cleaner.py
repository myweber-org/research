
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or np.array): The dataset.
    column (int): Index of the column to clean.
    
    Returns:
    np.array: Data with outliers removed.
    """
    data = np.array(data)
    col_data = data[:, column].astype(float)
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def example_usage():
    sample_data = [
        [1, 150.0],
        [2, 200.0],
        [3, 250.0],
        [4, 300.0],
        [5, 1000.0],
        [6, 50.0]
    ]
    
    print("Original data:")
    for row in sample_data:
        print(row)
    
    cleaned = remove_outliers_iqr(sample_data, column=1)
    
    print("\nCleaned data (outliers removed):")
    for row in cleaned:
        print(row)

if __name__ == "__main__":
    example_usage()
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Remove outliers using z-score
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
    print(f"Data cleaning complete. Cleaned data saved to {output_file}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            
            if col_max != col_min:
                normalized_df[col] = (dataframe[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def remove_missing_values(dataframe, strategy='drop', fill_value=None):
    """
    Handle missing values in dataframe
    """
    if strategy == 'drop':
        return dataframe.dropna()
    
    elif strategy == 'fill':
        if fill_value is None:
            fill_value = dataframe.mean(numeric_only=True)
        return dataframe.fillna(fill_value)
    
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Detect columns with significant skewness
    """
    skewed_cols = []
    
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(dataframe[col].dropna())
        if abs(skewness) > threshold:
            skewed_cols.append((col, skewness))
    
    return skewed_cols

def apply_log_transform(dataframe, columns):
    """
    Apply log transformation to specified columns
    """
    transformed_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            # Add small constant to handle zero values
            transformed_df[col] = np.log1p(dataframe[col])
    
    return transformed_df

def clean_dataset(dataframe, outlier_columns=None, normalize=True, handle_missing='drop'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_df = dataframe.copy()
    
    # Handle missing values
    cleaned_df = remove_missing_values(cleaned_df, strategy=handle_missing)
    
    # Remove outliers if specified
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    # Normalize numeric columns
    if normalize:
        cleaned_df = normalize_minmax(cleaned_df)
    
    return cleaned_df