import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing data in a DataFrame using specified strategy.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values. 
                       Options: 'mean', 'median', 'mode', 'drop', 'fill_zero'
        columns (list): List of columns to apply cleaning. If None, applies to all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    elif strategy == 'mean':
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif strategy == 'median':
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif strategy == 'mode':
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0)
    elif strategy == 'fill_zero':
        df_clean[columns] = df_clean[columns].fillna(0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean

def detect_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to check for outliers
        threshold (float): IQR multiplier threshold
    
    Returns:
        dict: Dictionary with outlier counts per column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_info = {}
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_info[col] = {
                'count': len(outliers),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_indices': outliers.index.tolist()
            }
    
    return outlier_info

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize data in specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to normalize
        method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df_normalized.columns:
            if method == 'minmax':
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val != min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                if std_val != 0:
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': [100, 200, 300, 400, 500],
        'category': ['X', 'Y', 'X', 'Y', 'Z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (mean imputation):")
    cleaned_df = clean_missing_data(df, strategy='mean')
    print(cleaned_df)
    
    print("\nOutlier detection:")
    outliers = detect_outliers_iqr(cleaned_df)
    for col, info in outliers.items():
        print(f"{col}: {info['count']} outliers")
    
    print("\nNormalized DataFrame (minmax):")
    normalized_df = normalize_data(cleaned_df.select_dtypes(include=[np.number]))
    print(normalized_df)
import pandas as pd
import numpy as np

def detect_outliers_iqr(dataframe, column):
    """
    Detect outliers in a specified column using the IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): The input dataframe
    column (str): Column name to check for outliers
    
    Returns:
    pd.DataFrame: DataFrame containing outlier rows
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
    
    return outliers

def remove_outliers_iqr(dataframe, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): The input dataframe
    column (str): Column name to remove outliers from
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    
    return filtered_data

def calculate_summary_statistics(dataframe, column):
    """
    Calculate summary statistics for a specified column.
    
    Parameters:
    dataframe (pd.DataFrame): The input dataframe
    column (str): Column name to calculate statistics for
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    stats = {
        'mean': dataframe[column].mean(),
        'median': dataframe[column].median(),
        'std': dataframe[column].std(),
        'min': dataframe[column].min(),
        'max': dataframe[column].max(),
        'count': dataframe[column].count(),
        'missing': dataframe[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(dataframe, columns=None):
    """
    Clean numeric data by removing NaN values and converting to appropriate types.
    
    Parameters:
    dataframe (pd.DataFrame): The input dataframe
    columns (list): List of column names to clean. If None, all numeric columns are cleaned.
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = dataframe.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    dataframe (pd.DataFrame): The input dataframe
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if dataframe.empty:
        return False, "DataFrame is empty"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"