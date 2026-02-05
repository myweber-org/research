def remove_duplicates(data_list):
    seen = set()
    result = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data_with_order(data_list, key=None):
    if key is None:
        key = lambda x: x
    seen = set()
    result = []
    for item in data_list:
        identifier = key(item)
        if identifier not in seen:
            seen.add(identifier)
            result.append(item)
    return result

if __name__ == "__main__":
    sample = [1, 2, 2, 3, 4, 4, 5]
    print(remove_duplicates(sample))
    
    sample_complex = [{"id": 1}, {"id": 2}, {"id": 1}, {"id": 3}]
    print(clean_data_with_order(sample_complex, key=lambda x: x["id"]))
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to process.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
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
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to analyze.
    
    Returns:
        dict: Dictionary containing summary statistics.
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

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[10, 'value'] = 500
    df.loc[20, 'value'] = -100
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal statistics:")
    stats = calculate_summary_statistics(df, 'value')
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    cleaned_stats = calculate_summary_statistics(cleaned_df, 'value')
    for key, value in cleaned_stats.items():
        print(f"{key}: {value:.2f}")
    
    removed_count = len(df) - len(cleaned_df)
    print(f"\nRemoved {removed_count} outliers")

if __name__ == "__main__":
    example_usage()import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.

    Args:
        dataframe: Input pandas DataFrame.
        subset: Column label or sequence of labels to consider for identifying duplicates.
                If None, all columns are used.
        keep: Determines which duplicates to mark.
              'first' : Mark duplicates as False except for the first occurrence.
              'last' : Mark duplicates as False except for the last occurrence.
              False : Mark all duplicates as True.

    Returns:
        DataFrame with duplicates removed.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(dataframe, column_name, fill_method='mean'):
    """
    Clean a numeric column by filling missing values.

    Args:
        dataframe: Input pandas DataFrame.
        column_name: Name of the column to clean.
        fill_method: Method to fill missing values ('mean', 'median', or 'zero').

    Returns:
        DataFrame with cleaned column.
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    if fill_method == 'mean':
        fill_value = dataframe[column_name].mean()
    elif fill_method == 'median':
        fill_value = dataframe[column_name].median()
    elif fill_method == 'zero':
        fill_value = 0
    else:
        raise ValueError("fill_method must be 'mean', 'median', or 'zero'")

    dataframe[column_name] = dataframe[column_name].fillna(fill_value)
    return dataframe