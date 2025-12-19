
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a column using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize a column using Min-Max scaling to range [0, 1].
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: Dataframe with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    data_copy = data.copy()
    min_val = data_copy[column].min()
    max_val = data_copy[column].max()
    
    if max_val == min_val:
        data_copy[f'{column}_normalized'] = 0.5
    else:
        data_copy[f'{column}_normalized'] = (data_copy[column] - min_val) / (max_val - min_val)
    
    return data_copy

def clean_dataset(data, numeric_columns, outlier_factor=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    outlier_factor (float): Factor for IQR outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_factor)
            cleaned_data = normalize_minmax(cleaned_data, column)
    
    return cleaned_data.reset_index(drop=True)

def validate_data(data, required_columns, min_rows=10):
    """
    Validate dataset structure and content.
    
    Parameters:
    data (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(data) < min_rows:
        return False, f"Dataset has fewer than {min_rows} rows"
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    return True, "Dataset is valid"

def get_summary_statistics(data, numeric_columns):
    """
    Generate summary statistics for numeric columns.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    summary = pd.DataFrame()
    
    for column in numeric_columns:
        if column in data.columns:
            col_stats = {
                'mean': data[column].mean(),
                'median': data[column].median(),
                'std': data[column].std(),
                'min': data[column].min(),
                'max': data[column].max(),
                'count': data[column].count(),
                'missing': data[column].isnull().sum()
            }
            summary[column] = pd.Series(col_stats)
    
    return summary.transpose()

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'score': np.random.uniform(0, 100, 100)
    })
    
    print("Original data shape:", sample_data.shape)
    
    cleaned = clean_dataset(sample_data, ['value', 'score'])
    print("Cleaned data shape:", cleaned.shape)
    
    is_valid, message = validate_data(cleaned, ['value', 'score'])
    print(f"Validation: {is_valid} - {message}")
    
    summary = get_summary_statistics(cleaned, ['value', 'score'])
    print("\nSummary statistics:")
    print(summary)