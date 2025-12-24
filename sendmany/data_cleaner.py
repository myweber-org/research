
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
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
        'count': len(df[column])
    }
    
    return stats

def main():
    # Example usage
    np.random.seed(42)
    data = {
        'values': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(200, 30, 10)
        ])
    }
    
    df = pd.DataFrame(data)
    print(f"Original data shape: {df.shape}")
    print(f"Original statistics: {calculate_summary_statistics(df, 'values')}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print(f"Cleaned statistics: {calculate_summary_statistics(cleaned_df, 'values')}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np

def clean_dataset(df, drop_threshold=0.5, normalize=True):
    """
    Clean a DataFrame by removing columns with excessive missing values,
    filling remaining missing values, and optionally normalizing numeric columns.
    """
    # Drop columns with missing values above threshold
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Fill remaining missing values with column median for numeric, mode for categorical
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype in [np.float64, np.int64]:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        else:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown')
    
    # Normalize numeric columns if requested
    if normalize:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_cleaned[col] = (df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std()
    
    return df_cleaned

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specified column using IQR or Z-score method.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        filtered_df = df[z_scores < threshold]
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return filtered_df.reset_index(drop=True)

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [np.nan, np.nan, np.nan, 4, 5, 6],
        'C': ['x', 'y', 'z', np.nan, 'x', 'y'],
        'D': [10.5, 20.3, 15.7, 18.2, 22.1, 19.8]
    })
    
    print("Original DataFrame:")
    print(sample_data)
    
    cleaned = clean_dataset(sample_data, drop_threshold=0.3)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    no_outliers = remove_outliers(cleaned, 'D', method='iqr')
    print("\nDataFrame after outlier removal:")
    print(no_outliers)