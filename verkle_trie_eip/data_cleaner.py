
import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif strategy == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Remove outliers using Z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / cleaned_df[numeric_cols].std())
    outlier_mask = (z_scores < threshold).all(axis=1)
    cleaned_df = cleaned_df[outlier_mask]
    
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if len(df) < min_rows:
        return False, f"Dataset must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"

def normalize_data(df, method='minmax'):
    """
    Normalize numeric columns in the dataset.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    method (str): Normalization method ('minmax', 'standard')
    
    Returns:
    pd.DataFrame: Normalized DataFrame
    """
    normalized_df = df.copy()
    numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
    
    if method == 'minmax':
        for col in numeric_cols:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    
    elif method == 'standard':
        for col in numeric_cols:
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            if std_val > 0:
                normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    
    return normalized_df

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned = clean_dataset(sample_data, strategy='mean')
    print("\nCleaned data:")
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")
    
    normalized = normalize_data(cleaned, method='minmax')
    print("\nNormalized data:")
    print(normalized)
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        data (pd.DataFrame): Input dataframe
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Args:
        data (pd.DataFrame): Input dataframe
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistics
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'count': data[column].count()
    }
    
    return stats

def clean_dataset(data, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Args:
        data (pd.DataFrame): Input dataframe
        columns_to_clean (list): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    cleaned_data = data.copy()
    
    for column in columns_to_clean:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_datadef remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data_with_order(input_list, key=None):
    if key is None:
        key = lambda x: x
    seen = set()
    result = []
    for item in input_list:
        identifier = key(item)
        if identifier not in seen:
            seen.add(identifier)
            result.append(item)
    return result

if __name__ == "__main__":
    sample = [1, 2, 2, 3, 4, 4, 5]
    print(remove_duplicates(sample))
    
    sample_dicts = [{"id": 1}, {"id": 2}, {"id": 1}, {"id": 3}]
    print(clean_data_with_order(sample_dicts, key=lambda x: x["id"]))