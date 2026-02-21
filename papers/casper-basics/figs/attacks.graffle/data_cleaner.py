
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_strategy='mean', drop_threshold=0.8):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    file_path (str): Path to the CSV file.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero').
    drop_threshold (float): Drop columns with missing ratio above this threshold.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {file_path}")
    
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if fill_strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif fill_strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif fill_strategy == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a new CSV file.
    
    Parameters:
    df (pandas.DataFrame): Cleaned DataFrame.
    output_path (str): Path to save the cleaned CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, fill_strategy='median', drop_threshold=0.7)
        save_cleaned_data(cleaned_df, output_file)
    except Exception as e:
        print(f"Error during data cleaning: {e}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers using IQR method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    """
    Standardize data using z-score.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns):
    """
    Apply cleaning operations to multiple numeric columns.
    """
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
            cleaned_df = standardize_zscore(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['X', 'Y', 'Z'], 200)
    })
    numeric_cols = ['feature_a', 'feature_b']
    result = clean_dataset(sample_data, numeric_cols)
    print(result.head())
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {result.shape}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, factor=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def zscore_normalize(dataframe, column):
    """
    Normalize column using z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    
    if std_val > 0:
        dataframe[column + '_normalized'] = (dataframe[column] - mean_val) / std_val
    else:
        dataframe[column + '_normalized'] = 0
    
    return dataframe

def minmax_normalize(dataframe, column, feature_range=(0, 1)):
    """
    Normalize column using min-max scaling.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to normalize
        feature_range: Desired range of transformed data
    
    Returns:
        DataFrame with normalized column
    """
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if max_val > min_val:
        a, b = feature_range
        dataframe[column + '_scaled'] = a + ((dataframe[column] - min_val) * (b - a)) / (max_val - min_val)
    else:
        dataframe[column + '_scaled'] = a
    
    return dataframe

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Detect columns with significant skewness.
    
    Args:
        dataframe: pandas DataFrame
        threshold: Absolute skewness threshold
    
    Returns:
        Dictionary of skewed columns and their skewness values
    """
    skewed_cols = {}
    
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        skewness = dataframe[col].skew()
        if abs(skewness) > threshold:
            skewed_cols[col] = skewness
    
    return skewed_cols

def log_transform(dataframe, column):
    """
    Apply log transformation to reduce skewness.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to transform
    
    Returns:
        DataFrame with transformed column
    """
    if dataframe[column].min() <= 0:
        offset = abs(dataframe[column].min()) + 1
        dataframe[column + '_log'] = np.log(dataframe[column] + offset)
    else:
        dataframe[column + '_log'] = np.log(dataframe[column])
    
    return dataframe

def clean_dataset(dataframe, numeric_columns=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame
        numeric_columns: List of numeric columns to process
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = zscore_normalize(cleaned_df, col)
    
    return cleaned_df

def calculate_statistics(dataframe):
    """
    Calculate comprehensive statistics for numeric columns.
    
    Args:
        dataframe: pandas DataFrame
    
    Returns:
        DataFrame with statistics
    """
    stats_df = pd.DataFrame()
    
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        col_stats = {
            'mean': dataframe[col].mean(),
            'median': dataframe[col].median(),
            'std': dataframe[col].std(),
            'min': dataframe[col].min(),
            'max': dataframe[col].max(),
            'skewness': dataframe[col].skew(),
            'kurtosis': dataframe[col].kurtosis(),
            'missing': dataframe[col].isnull().sum()
        }
        stats_df[col] = pd.Series(col_stats)
    
    return stats_df
def remove_duplicates(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result