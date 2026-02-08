
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_column_zscore(dataframe, column):
    """
    Normalize a column using Z-score normalization.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result_df = dataframe.copy()
    mean_val = result_df[column].mean()
    std_val = result_df[column].std()
    
    if std_val > 0:
        result_df[f"{column}_normalized"] = (result_df[column] - mean_val) / std_val
    else:
        result_df[f"{column}_normalized"] = 0
    
    return result_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to process, None for all numeric columns
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    result_df = dataframe.copy()
    
    if columns is None:
        columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in result_df.columns:
            continue
            
        if strategy == 'drop':
            result_df = result_df.dropna(subset=[col])
        elif strategy == 'mean':
            result_df[col].fillna(result_df[col].mean(), inplace=True)
        elif strategy == 'median':
            result_df[col].fillna(result_df[col].median(), inplace=True)
        elif strategy == 'mode':
            result_df[col].fillna(result_df[col].mode()[0], inplace=True)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return result_df

def create_data_summary(dataframe):
    """
    Create a summary statistics DataFrame.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    summary_data = {
        'mean': dataframe[numeric_cols].mean(),
        'median': dataframe[numeric_cols].median(),
        'std': dataframe[numeric_cols].std(),
        'min': dataframe[numeric_cols].min(),
        'max': dataframe[numeric_cols].max(),
        'missing': dataframe[numeric_cols].isnull().sum(),
        'unique': dataframe[numeric_cols].nunique()
    }
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    dataframe (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', threshold=1.5):
    """
    Clean dataset by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    missing_strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for outlier detection ('iqr', 'zscore')
    threshold (float): Threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    elif missing_strategy == 'median':
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    elif missing_strategy == 'mode':
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    elif missing_strategy == 'drop':
        df_clean = df_clean.dropna()
    
    # Handle outliers for numeric columns only
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if outlier_method == 'iqr':
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Cap outliers
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    elif outlier_method == 'zscore':
        for col in numeric_cols:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            mask = z_scores > threshold
            df_clean.loc[mask, col] = df_clean[col].mean()
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"Dataframe has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataframe is valid"

def get_data_summary(df):
    """
    Generate summary statistics for dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    for col in categorical_cols:
        summary['categorical_stats'][col] = {
            'unique_values': df[col].nunique(),
            'top_value': df[col].mode().iloc[0] if not df[col].mode().empty else None,
            'top_count': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
        }
    
    return summary

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'x', 'y', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Clean the data
    df_clean = clean_dataset(df, missing_strategy='mean', outlier_method='iqr')
    print("Cleaned DataFrame:")
    print(df_clean)
    print()
    
    # Validate
    is_valid, message = validate_dataframe(df_clean, required_columns=['A', 'B', 'C'])
    print(f"Validation: {is_valid} - {message}")
    print()
    
    # Get summary
    summary = get_data_summary(df_clean)
    print("Data Summary:")
    print(f"Shape: {summary['shape']}")
    print(f"Numeric Stats: {summary['numeric_stats']}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers from a pandas Series using the IQR method.
    
    Parameters:
    data (pd.Series): Input data series
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.Series: Data with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    series = data[column]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(series >= lower_bound) & (series <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
    Parameters:
    data (pd.Series): Input data series
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized data
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    series = data[column]
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    
    normalized = (series - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Parameters:
    data (pd.Series): Input data series
    column (str): Column name to standardize
    
    Returns:
    pd.Series: Standardized data with mean=0, std=1
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    series = data[column]
    mean_val = series.mean()
    std_val = series.std()
    
    if std_val == 0:
        return pd.Series([0] * len(series), index=series.index)
    
    standardized = (series - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to process
    outlier_method (str): Method for outlier removal ('iqr' or None)
    normalize_method (str): Normalization method ('minmax', 'zscore', or None)
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    series = df[column]
    stats_dict = {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'count': series.count(),
        'missing': series.isnull().sum()
    }
    
    return stats_dict

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("\nOriginal statistics for feature_a:")
    print(calculate_statistics(sample_data, 'feature_a'))
    
    cleaned_data = clean_dataset(
        sample_data, 
        numeric_columns=['feature_a', 'feature_b'],
        outlier_method='iqr',
        normalize_method='minmax'
    )
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("\nCleaned statistics for feature_a:")
    print(calculate_statistics(cleaned_data, 'feature_a'))