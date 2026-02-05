import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default True.
    fill_missing (bool): Whether to fill missing values. Default False.
    fill_value: Value to use for filling missing data. Default 0.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
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
import numpy as np
import pandas as pd

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
    
    return filtered_df

def clean_dataset(df, columns=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    return cleaned_df
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    return filtered_df

def z_score_normalize(dataframe, columns):
    """
    Normalize specified columns using z-score normalization
    """
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        
        mean_val = normalized_df[col].mean()
        std_val = normalized_df[col].std()
        
        if std_val > 0:
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        else:
            normalized_df[col] = 0
    
    return normalized_df

def min_max_normalize(dataframe, columns, feature_range=(0, 1)):
    """
    Normalize specified columns using min-max normalization
    """
    normalized_df = dataframe.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        col_range = col_max - col_min
        
        if col_range > 0:
            normalized_df[col] = ((normalized_df[col] - col_min) / col_range) * (max_val - min_val) + min_val
        else:
            normalized_df[col] = min_val
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy
    """
    cleaned_df = dataframe.copy()
    
    if columns is None:
        columns = cleaned_df.columns
    
    for col in columns:
        if col not in cleaned_df.columns:
            continue
            
        if cleaned_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = cleaned_df[col].mean()
            elif strategy == 'median':
                fill_value = cleaned_df[col].median()
            elif strategy == 'mode':
                fill_value = cleaned_df[col].mode()[0]
            elif strategy == 'drop':
                cleaned_df = cleaned_df.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            cleaned_df[col] = cleaned_df[col].fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(dataframe) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def create_sample_data():
    """
    Create sample data for testing
    """
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.uniform(0, 1, 100),
        'feature_c': np.random.exponential(2, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(100, 5), 'feature_a'] = np.nan
    df.loc[np.random.choice(100, 3), 'feature_b'] = np.nan
    
    return dfimport pandas as pd
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
        'count': df[column].count()
    }
    
    return stats

def process_dataframe(df, numeric_columns):
    """
    Process DataFrame by removing outliers from multiple numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of column names to process
    
    Returns:
        pd.DataFrame: Processed DataFrame with outliers removed
    """
    processed_df = df.copy()
    
    for column in numeric_columns:
        if column in processed_df.columns and pd.api.types.is_numeric_dtype(processed_df[column]):
            processed_df = remove_outliers_iqr(processed_df, column)
    
    return processed_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1100]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nSummary statistics for temperature:")
    print(calculate_summary_statistics(df, 'temperature'))
    
    processed_df = process_dataframe(df, ['temperature', 'pressure'])
    print("\nProcessed DataFrame (outliers removed):")
    print(processed_df)
    print("\nSummary statistics for temperature after processing:")
    print(calculate_summary_statistics(processed_df, 'temperature'))