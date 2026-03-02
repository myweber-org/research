import numpy as np
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
    return Trueimport pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for missing value imputation ('mean', 'median', 'mode')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    for column in cleaned_df.columns:
        if cleaned_df[column].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            elif strategy == 'median':
                cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            elif strategy == 'mode':
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
        else:
            # For categorical columns, fill with most frequent value
            cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    # Handle outliers using Z-score method for numerical columns
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        z_scores = np.abs((cleaned_df[column] - cleaned_df[column].mean()) / cleaned_df[column].std())
        outliers = z_scores > outlier_threshold
        
        if outliers.any():
            # Cap outliers at threshold * standard deviation
            upper_bound = cleaned_df[column].mean() + outlier_threshold * cleaned_df[column].std()
            lower_bound = cleaned_df[column].mean() - outlier_threshold * cleaned_df[column].std()
            
            cleaned_df.loc[cleaned_df[column] > upper_bound, column] = upper_bound
            cleaned_df.loc[cleaned_df[column] < lower_bound, column] = lower_bound
    
    return cleaned_df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list): Columns to consider for duplicate detection
    keep (str): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize numerical columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to normalize (None for all numerical columns)
    method (str): Normalization method ('minmax', 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    normalized_df = df.copy()
    
    if columns is None:
        columns = normalized_df.select_dtypes(include=[np.number]).columns
    
    for column in columns:
        if column in normalized_df.columns and normalized_df[column].dtype in ['int64', 'float64']:
            if method == 'minmax':
                min_val = normalized_df[column].min()
                max_val = normalized_df[column].max()
                if max_val != min_val:
                    normalized_df[column] = (normalized_df[column] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = normalized_df[column].mean()
                std_val = normalized_df[column].std()
                if std_val != 0:
                    normalized_df[column] = (normalized_df[column] - mean_val) / std_val
    
    return normalized_df

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 60],
        'C': ['a', 'b', 'a', 'c', 'b', 'a']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned = clean_dataset(df, strategy='median', outlier_threshold=2)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    # Normalize the data
    normalized = normalize_data(cleaned, method='minmax')
    print("Normalized DataFrame:")
    print(normalized)