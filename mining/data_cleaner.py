
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns=None):
    """
    Remove outliers from DataFrame using Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to process. If None, process all numeric columns.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def calculate_statistics(df, columns=None):
    """
    Calculate basic statistics for numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to process. If None, process all numeric columns.
    
    Returns:
        pd.DataFrame: Statistics summary
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = {}
    for col in columns:
        if col in df.columns:
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'count': df[col].count(),
                'missing': df[col].isnull().sum()
            }
    
    return pd.DataFrame(stats).T

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original shape:", df.shape)
    print("Original statistics:")
    print(calculate_statistics(df))
    
    df_clean = remove_outliers_iqr(df, ['A', 'B'])
    
    print("\nCleaned shape:", df_clean.shape)
    print("Cleaned statistics:")
    print(calculate_statistics(df_clean))