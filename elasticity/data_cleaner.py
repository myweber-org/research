
import pandas as pd
import numpy as np

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

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[0, 'A'] = 1000
    df.loc[1, 'B'] = 500
    
    print(f"Original dataset shape: {df.shape}")
    cleaned_df = clean_dataset(df)
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistics
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'count': data[column].count()
    }
    return stats

def clean_dataset(data, numeric_columns):
    """
    Clean dataset by removing outliers from multiple numeric columns.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            original_count = len(cleaned_data)
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
            removed_count = original_count - len(cleaned_data)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_dataimport pandas as pd
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
    df_clean = df_clean[outlier_mask]
    
    return df_clean.reset_index(drop=True)

def validate_data(df, required_columns, min_rows=10):
    """
    Validate dataset structure and content.
    """
    if len(df) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows")
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'feature_a': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'feature_b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned_data = clean_dataset(sample_data, numeric_columns=['feature_a', 'feature_b'])
    print("\nCleaned data:")
    print(cleaned_data)
    
    try:
        validate_data(cleaned_data, required_columns=['feature_a', 'feature_b'])
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """Fill missing values using specified strategy."""
    if columns is None:
        columns = df.columns
    
    df_filled = df.copy()
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_filled[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'zero':
                df_filled[col] = df[col].fillna(0)
        else:
            df_filled[col] = df[col].fillna('Unknown')
    
    return df_filled

def normalize_column(df, column, method='minmax'):
    """Normalize specified column using given method."""
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df[column] = (df[column] - mean_val) / std_val
    
    return df

def clean_dataframe(df, operations=None):
    """Apply multiple cleaning operations to DataFrame."""
    if operations is None:
        operations = ['remove_duplicates', 'fill_missing']
    
    cleaned_df = df.copy()
    
    if 'remove_duplicates' in operations:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if 'fill_missing' in operations:
        cleaned_df = fill_missing_values(cleaned_df)
    
    return cleaned_df

def validate_dataframe(df, rules=None):
    """Validate DataFrame against specified rules."""
    if rules is None:
        rules = {}
    
    validation_results = {}
    
    for column, rule in rules.items():
        if column in df.columns:
            if 'min' in rule:
                validation_results[f'{column}_min'] = df[column].min() >= rule['min']
            if 'max' in rule:
                validation_results[f'{column}_max'] = df[column].max() <= rule['max']
            if 'unique' in rule:
                validation_results[f'{column}_unique'] = df[column].nunique() == rule['unique']
    
    return validation_results