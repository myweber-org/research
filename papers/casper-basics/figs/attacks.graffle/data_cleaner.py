
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or np.array): Input data
    column (int): Index of column to process (if 2D array) or ignored (if 1D array)
    
    Returns:
    np.array: Data with outliers removed
    """
    data_array = np.array(data)
    
    if data_array.ndim == 2:
        column_data = data_array[:, column]
    else:
        column_data = data_array
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if data_array.ndim == 2:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        return data_array[mask]
    else:
        return column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (list or np.array): Input data
    
    Returns:
    dict: Dictionary containing mean, median, std, min, max
    """
    data_array = np.array(data)
    
    return {
        'mean': np.mean(data_array),
        'median': np.median(data_array),
        'std': np.std(data_array),
        'min': np.min(data_array),
        'max': np.max(data_array)
    }

if __name__ == "__main__":
    sample_data = [10, 12, 13, 15, 16, 18, 20, 22, 24, 100]
    
    cleaned_data = remove_outliers_iqr(sample_data, 0)
    stats = calculate_statistics(cleaned_data)
    
    print(f"Original data: {sample_data}")
    print(f"Cleaned data: {cleaned_data}")
    print(f"Statistics: {stats}")def clean_data(data):
    """
    Remove duplicate entries from a list and sort the remaining items.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list")
    # Remove duplicates by converting to a set, then back to a list
    unique_data = list(set(data))
    # Sort the list
    unique_data.sort()
    return unique_data
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
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
    
    Parameters:
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
            df_copy[f'{column}_normalized'] = 0
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    else:
        raise ValueError("Method must be either 'minmax' or 'zscore'")
    
    return df_copy

if __name__ == "__main__":
    # Example usage
    sample_data = {'values': [1, 2, 3, 4, 5, 100, 200, 300]}
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("DataFrame after removing outliers:")
    print(cleaned_df)
    print()
    
    stats = calculate_summary_statistics(df, 'values')
    print("Summary statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()
    
    normalized_df = normalize_column(df, 'values', method='minmax')
    print("DataFrame with normalized column:")
    print(normalized_df)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default True.
    fill_missing (str): Strategy to fill missing values. 
                       Options: 'mean', 'median', 'mode', or 'drop'. Default 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (bool, str) indicating success and message.
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    method (str): Outlier detection method ('iqr' or 'zscore').
    threshold (float): Threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    elif method == 'zscore':
        from scipy import stats
        z_scores = stats.zscore(df[column].dropna())
        mask = abs(z_scores) < threshold
        mask = pd.Series(mask, index=df[column].dropna().index)
        mask = mask.reindex(df.index).fillna(True)
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or array-like): The dataset containing the column to clean.
    column (int or str): The index or name of the column to process.
    
    Returns:
    tuple: A tuple containing:
        - cleaned_data (numpy array): Data with outliers removed.
        - outlier_indices (numpy array): Indices of removed outliers.
    """
    data_array = np.array(data)
    col_data = data_array[:, column] if isinstance(column, int) else data_array[column]
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
    cleaned_data = data_array[~outlier_mask]
    
    return cleaned_data, np.where(outlier_mask)[0]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (numpy array): The cleaned dataset.
    column (int): The index of the column to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, std, min, and max values.
    """
    col_data = data[:, column]
    
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data),
        'min': np.min(col_data),
        'max': np.max(col_data),
        'count': len(col_data)
    }
    
    return stats

def clean_dataset(data, columns_to_clean):
    """
    Main function to clean multiple columns in a dataset.
    
    Parameters:
    data (list or array-like): The original dataset.
    columns_to_clean (list): List of column indices to clean.
    
    Returns:
    dict: Dictionary containing cleaned data and cleaning report.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    cleaning_report = {}
    current_data = data.copy()
    
    for col in columns_to_clean:
        cleaned_data, outliers = remove_outliers_iqr(current_data, col)
        stats = calculate_statistics(cleaned_data, col)
        
        cleaning_report[col] = {
            'outliers_removed': len(outliers),
            'outlier_indices': outliers.tolist(),
            'statistics': stats,
            'original_count': len(current_data),
            'cleaned_count': len(cleaned_data)
        }
        
        current_data = cleaned_data
    
    return {
        'cleaned_data': current_data,
        'report': cleaning_report
    }

if __name__ == "__main__":
    # Example usage
    sample_data = np.random.randn(100, 3) * 10 + 50
    sample_data[10, 0] = 200  # Add an outlier
    sample_data[20, 1] = -100  # Add another outlier
    
    result = clean_dataset(sample_data, [0, 1, 2])
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Cleaned data shape: {result['cleaned_data'].shape}")
    
    for col, report in result['report'].items():
        print(f"\nColumn {col}:")
        print(f"  Outliers removed: {report['outliers_removed']}")
        print(f"  Original count: {report['original_count']}")
        print(f"  Cleaned count: {report['cleaned_count']}")
        print(f"  Mean: {report['statistics']['mean']:.2f}")
        print(f"  Std: {report['statistics']['std']:.2f}")