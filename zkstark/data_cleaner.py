import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Index of column to clean
    
    Returns:
    numpy.ndarray: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    column_data = data[:, column]
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Index of column to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    cleaned_data = remove_outliers_iqr(data, column)
    column_cleaned = cleaned_data[:, column]
    
    stats = {
        'mean': np.mean(column_cleaned),
        'median': np.median(column_cleaned),
        'std': np.std(column_cleaned),
        'min': np.min(column_cleaned),
        'max': np.max(column_cleaned),
        'original_size': len(data),
        'cleaned_size': len(cleaned_data),
        'removed_count': len(data) - len(cleaned_data)
    }
    
    return stats

def example_usage():
    """
    Example demonstrating the usage of data cleaning functions.
    """
    np.random.seed(42)
    
    sample_data = np.random.randn(1000, 3)
    sample_data[:, 1] = sample_data[:, 1] * 10 + 50
    
    outliers = np.random.randn(50, 3) * 20 + 100
    sample_data = np.vstack([sample_data, outliers])
    
    print(f"Original data shape: {sample_data.shape}")
    
    cleaned = remove_outliers_iqr(sample_data, 1)
    print(f"Cleaned data shape: {cleaned.shape}")
    
    stats = calculate_statistics(sample_data, 1)
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    example_usage()import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_method='iqr', fill_value=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'constant')
    outlier_method (str): Method for outlier detection ('iqr', 'zscore')
    fill_value: Value to use when strategy='constant'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    # Handle missing values
    if strategy == 'mean':
        df_clean = df_clean.fillna(df_clean.mean())
    elif strategy == 'median':
        df_clean = df_clean.fillna(df_clean.median())
    elif strategy == 'mode':
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    elif strategy == 'constant' and fill_value is not None:
        df_clean = df_clean.fillna(fill_value)
    
    # Handle outliers
    if outlier_method == 'iqr':
        for column in df_clean.select_dtypes(include=[np.number]).columns:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            df_clean[column] = np.where(df_clean[column] < lower_bound, lower_bound, df_clean[column])
            df_clean[column] = np.where(df_clean[column] > upper_bound, upper_bound, df_clean[column])
    
    elif outlier_method == 'zscore':
        for column in df_clean.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
            df_clean[column] = np.where(z_scores > 3, df_clean[column].median(), df_clean[column])
    
    return df_clean

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset: Columns to consider for duplicates
    keep: Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: DataFrame without duplicates
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_data(df, method='minmax'):
    """
    Normalize numerical columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    method (str): Normalization method ('minmax', 'standard')
    
    Returns:
    pd.DataFrame: Normalized DataFrame
    """
    df_norm = df.copy()
    
    numerical_cols = df_norm.select_dtypes(include=[np.number]).columns
    
    if method == 'minmax':
        for col in numerical_cols:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    
    elif method == 'standard':
        for col in numerical_cols:
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val > 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    # Clean the data
    cleaned_df = clean_dataset(df, strategy='median', outlier_method='iqr')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Normalize the data
    normalized_df = normalize_data(cleaned_df, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized_df)import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val == min_val:
        return df[column].apply(lambda x: 0.0)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True