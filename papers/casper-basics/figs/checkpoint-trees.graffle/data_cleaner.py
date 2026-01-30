
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