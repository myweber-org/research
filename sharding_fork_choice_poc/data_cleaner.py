
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
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to normalize
        method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
        else:
            df_copy[f'{column}_normalized'] = 0.5
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
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
    
    return filtered_df

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list, optional): List of numeric columns to clean.
                                         If None, uses all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(100),
        'value': np.random.randn(100) * 10 + 50
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[95:99, 'value'] = [200, -100, 300, 150, 250]
    
    print(f"Original dataset shape: {df.shape}")
    cleaned_df = clean_dataset(df, ['value'])
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
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
    
    return filtered_df

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistics
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

def main():
    # Example usage
    np.random.seed(42)
    data = {
        'values': np.random.normal(100, 15, 1000).tolist() + [500, -200]  # Add some outliers
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_basic_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_basic_stats(cleaned_df, 'values'))
    
    removed_count = len(df) - len(cleaned_df)
    print(f"\nRemoved {removed_count} outliers")

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def z_score_normalize(data, column):
    """
    Normalize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data
    
    normalized_data = data.copy()
    normalized_data[f'{column}_normalized'] = (data[column] - mean_val) / std_val
    return normalized_data

def min_max_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
        feature_range: tuple of (min, max) for output range
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data
    
    normalized_data = data.copy()
    normalized_data[f'{column}_scaled'] = (
        (data[column] - min_val) / (max_val - min_val) * 
        (feature_range[1] - feature_range[0]) + feature_range[0]
    )
    return normalized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process, None for all numeric columns
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    processed_data = data.copy()
    
    for column in columns:
        if column not in processed_data.columns:
            continue
            
        if strategy == 'drop':
            processed_data = processed_data.dropna(subset=[column])
        elif strategy == 'mean':
            processed_data[column] = processed_data[column].fillna(processed_data[column].mean())
        elif strategy == 'median':
            processed_data[column] = processed_data[column].fillna(processed_data[column].median())
        elif strategy == 'mode':
            processed_data[column] = processed_data[column].fillna(processed_data[column].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return processed_data

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5, 
                  normalize_method='zscore', missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_multiplier: IQR multiplier for outlier detection
        normalize_method: normalization method ('zscore', 'minmax', or None)
        missing_strategy: strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Handle missing values
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy, 
                                         columns=numeric_columns)
    
    # Remove outliers
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_multiplier)
    
    # Normalize data
    if normalize_method == 'zscore':
        for column in numeric_columns:
            if column in cleaned_data.columns:
                cleaned_data = z_score_normalize(cleaned_data, column)
    elif normalize_method == 'minmax':
        for column in numeric_columns:
            if column in cleaned_data.columns:
                cleaned_data = min_max_normalize(cleaned_data, column)
    
    return cleaned_data

def get_data_statistics(data, columns=None):
    """
    Get statistical summary of the data.
    
    Args:
        data: pandas DataFrame
        columns: list of columns to analyze, None for all numeric columns
    
    Returns:
        Dictionary with statistical metrics
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    stats_dict = {}
    
    for column in columns:
        if column not in data.columns:
            continue
            
        col_data = data[column].dropna()
        if len(col_data) == 0:
            continue
            
        stats_dict[column] = {
            'count': len(col_data),
            'mean': col_data.mean(),
            'std': col_data.std(),
            'min': col_data.min(),
            '25%': col_data.quantile(0.25),
            'median': col_data.median(),
            '75%': col_data.quantile(0.75),
            'max': col_data.max(),
            'skewness': col_data.skew(),
            'kurtosis': col_data.kurtosis(),
            'missing_values': data[column].isnull().sum()
        }
    
    return stats_dict

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Add some outliers and missing values
    sample_data.loc[10, 'feature1'] = 500
    sample_data.loc[20, 'feature2'] = 1000
    sample_data.loc[30:40, 'feature3'] = np.nan
    
    print("Original data shape:", sample_data.shape)
    print("\nOriginal statistics:")
    stats = get_data_statistics(sample_data)
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        for key, value in col_stats.items():
            print(f"  {key}: {value:.4f}")
    
    # Clean the data
    cleaned = clean_dataset(sample_data, normalize_method='zscore')
    
    print("\n\nCleaned data shape:", cleaned.shape)
    print("\nCleaned statistics:")
    cleaned_stats = get_data_statistics(cleaned)
    for col, col_stats in cleaned_stats.items():
        if 'normalized' in col or 'scaled' in col:
            print(f"\n{col}:")
            for key, value in col_stats.items():
                print(f"  {key}: {value:.4f}")