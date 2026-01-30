
import pandas as pd
import numpy as np

def clean_data(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                else:  # mode
                    fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in '{col}' with {fill_missing}: {fill_value:.2f}")
    
    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10.5, 20.3, np.nan, 30.1, np.nan, 40.7],
        'category': ['A', 'B', 'A', 'A', 'C', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_data(df, drop_duplicates=True, fill_missing='mean')
    
    print("\nCleaned data:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['id', 'value', 'category'], min_rows=3)
    print(f"\nData is valid: {is_valid}")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. 
            Can be 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    if fill_missing is not None:
        if fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        elif isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        else:
            raise ValueError("fill_missing must be 'mean', 'median', 'mode', or a dictionary.")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Filled {missing_count} missing values.")
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for required columns and minimum row count.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if len(df) < min_rows:
        print(f"Dataset has only {len(df)} rows, minimum required is {min_rows}.")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None],
        'B': [4, None, 6, 6, 8],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (drop duplicates, fill with mean):")
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print(cleaned)
    
    print("\nValidation result:", validate_dataset(cleaned, required_columns=['A', 'B'], min_rows=2))
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
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

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    stats = {
        'original_count': len(df),
        'cleaned_count': len(remove_outliers_iqr(df, column)),
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.array([300, 350, -50, 400])
        ])
    }
    
    df = pd.DataFrame(data)
    print(f"Original data shape: {df.shape}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print(f"Cleaned data shape: {cleaned_df.shape}")
    
    stats = calculate_summary_stats(df, 'values')
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    example_usage()import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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

def clean_dataset(file_path, output_path=None):
    """
    Clean a dataset by removing outliers from numeric columns.
    
    Args:
        file_path (str): Path to input CSV file
        output_path (str, optional): Path to save cleaned data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    
    original_rows = len(df)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
    
    removed_rows = original_rows - len(df)
    print(f"Cleaning complete. Removed {removed_rows} outliers.")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_data = clean_dataset(input_file, output_file)
    
    if cleaned_data is not None:
        print(f"Original data shape: (loaded from {input_file})")
        print(f"Cleaned data shape: {cleaned_data.shape}")
        print("\nSummary statistics:")
        print(cleaned_data.describe())
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Args:
        df: Input pandas DataFrame
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'drop')
        outlier_threshold: Z-score threshold for outlier detection
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Remove outliers using Z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
        cleaned_df = cleaned_df[z_scores < outlier_threshold]
    
    return cleaned_df.reset_index(drop=True)

def normalize_data(df, method='minmax'):
    """
    Normalize numeric columns in the DataFrame.
    
    Args:
        df: Input pandas DataFrame
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        Normalized pandas DataFrame
    """
    normalized_df = df.copy()
    numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
    
    if method == 'minmax':
        for col in numeric_cols:
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            if col_max != col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
    
    elif method == 'zscore':
        for col in numeric_cols:
            col_mean = normalized_df[col].mean()
            col_std = normalized_df[col].std()
            if col_std != 0:
                normalized_df[col] = (normalized_df[col] - col_mean) / col_std
    
    return normalized_df

def validate_dataframe(df):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: Input pandas DataFrame
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_results
import pandas as pd
import re

def clean_string_column(series, case='lower', strip=True, remove_extra_spaces=True):
    """
    Standardize string values in a pandas Series.
    """
    if not pd.api.types.is_string_dtype(series):
        series = series.astype(str)
    
    if case == 'lower':
        series = series.str.lower()
    elif case == 'upper':
        series = series.str.upper()
    elif case == 'title':
        series = series.str.title()
    
    if strip:
        series = series.str.strip()
    
    if remove_extra_spaces:
        series = series.apply(lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else x)
    
    return series

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame with optional subset.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def standardize_missing_values(df, missing_values=['null', 'none', 'na', 'nan', ''], replacement=pd.NA):
    """
    Replace common missing value representations with pandas NA.
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].replace(missing_values, replacement)
    return df

def clean_dataframe(df, string_columns=None, case='lower', remove_dups=True, standardize_missing=True):
    """
    Apply multiple cleaning operations to a DataFrame.
    """
    df = df.copy()
    
    if standardize_missing:
        df = standardize_missing_values(df)
    
    if string_columns is None:
        string_columns = df.select_dtypes(include=['object']).columns
    
    for col in string_columns:
        if col in df.columns:
            df[col] = clean_string_column(df[col], case=case)
    
    if remove_dups:
        df = remove_duplicates(df)
    
    return df