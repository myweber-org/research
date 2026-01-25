
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna('Unknown')
        
        print("Filled missing values in numeric and categorical columns")
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame for common data quality issues.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    
    Returns:
    dict: Dictionary containing validation results.
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

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataframe(df))
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Validation Results:")
    print(validate_dataframe(cleaned_df))import pandas as pd
import numpy as np

def clean_csv_data(file_path, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and dropping specified columns.
    
    Parameters:
    file_path (str): Path to the CSV file.
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop').
    columns_to_drop (list): List of column names to drop from the dataset.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
    
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy in ['mean', 'median']:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if missing_strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        else:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif missing_strategy == 'mode':
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
    
    return df

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    column (str): Column name to check for outliers.
    
    Returns:
    pandas.Series: Boolean series indicating outliers.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    column (str): Column name to normalize.
    method (str): Normalization method ('minmax' or 'zscore').
    
    Returns:
    pandas.DataFrame: DataFrame with normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, np.nan, 50],
        'C': [100, 200, 300, 400, 500]
    })
    
    cleaned = clean_csv_data('sample.csv', missing_strategy='mean')
    outliers = detect_outliers_iqr(cleaned, 'A')
    normalized = normalize_column(cleaned, 'B', method='zscore')
    
    print("Data cleaning completed successfully.")
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Columns to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def main():
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David'],
        'score': ['95', '88', '88', '92', 'invalid', '85']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    # Remove duplicates
    df_clean = remove_duplicates(df, subset=['id', 'name'])
    
    # Clean numeric column
    df_clean = clean_numeric_columns(df_clean, ['score'])
    
    print("\nCleaned DataFrame:")
    print(df_clean)
    
    # Validate
    is_valid, message = validate_dataframe(df_clean, required_columns=['id', 'name'])
    print(f"\nValidation: {message}")

if __name__ == "__main__":
    main()