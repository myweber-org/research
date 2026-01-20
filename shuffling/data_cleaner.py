import pandas as pd
import numpy as np
from scipy import stats

def clean_dataset(df, numeric_columns=None, z_threshold=3):
    """
    Clean dataset by handling missing values, normalizing numeric columns,
    and removing outliers based on z-score.
    """
    df_clean = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Handle missing values
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Normalize numeric columns
    for col in numeric_columns:
        if col in df_clean.columns:
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            if std_val > 0:
                df_clean[col] = (df_clean[col] - mean_val) / std_val
    
    # Remove outliers using z-score
    z_scores = np.abs(stats.zscore(df_clean[numeric_columns]))
    outlier_mask = (z_scores < z_threshold).all(axis=1)
    df_clean = df_clean[outlier_mask].reset_index(drop=True)
    
    return df_clean

def validate_data(df, required_columns):
    """
    Validate that required columns exist and have no null values.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found in columns: {null_counts[null_counts > 0].to_dict()}")
    
    return True

def sample_data(df, sample_size=1000, random_state=42):
    """
    Sample data from dataframe while maintaining distribution.
    """
    if len(df) <= sample_size:
        return df
    
    return df.sample(n=sample_size, random_state=random_state)
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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

def process_numerical_data(df, columns=None):
    """
    Process multiple numerical columns by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, processes all numerical columns.
    
    Returns:
    pd.DataFrame: Processed DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    processed_df = df.copy()
    
    for column in columns:
        if column in processed_df.columns and pd.api.types.is_numeric_dtype(processed_df[column]):
            try:
                processed_df = remove_outliers_iqr(processed_df, column)
            except Exception as e:
                print(f"Error processing column '{column}': {e}")
    
    return processed_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original summary statistics:")
    for col in df.columns:
        stats = calculate_summary_statistics(df, col)
        print(f"{col}: {stats}")
    
    cleaned_df = process_numerical_data(df)
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary statistics:")
    for col in cleaned_df.columns:
        stats = calculate_summary_statistics(cleaned_df, col)
        print(f"{col}: {stats}")