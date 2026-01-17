
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a dataset by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        columns_to_check: list of columns to check for duplicates (default: all columns)
        fill_missing: strategy to fill missing values ('mean', 'median', 'mode', or value)
    
    Returns:
        Cleaned pandas DataFrame
    """
    original_rows = df.shape[0]
    
    # Remove duplicates
    if columns_to_check is None:
        df_clean = df.drop_duplicates()
    else:
        df_clean = df.drop_duplicates(subset=columns_to_check)
    
    removed_duplicates = original_rows - df_clean.shape[0]
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if col in numeric_cols:
                if fill_missing == 'mean':
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif fill_missing == 'median':
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif fill_missing == 'mode':
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                else:
                    try:
                        fill_value = float(fill_missing)
                        df_clean[col].fillna(fill_value, inplace=True)
                    except ValueError:
                        df_clean[col].fillna(0, inplace=True)
            else:
                # For categorical columns, fill with most frequent value
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
    
    missing_filled = df.isnull().sum().sum() - df_clean.isnull().sum().sum()
    
    print(f"Removed {removed_duplicates} duplicate rows")
    print(f"Filled {missing_filled} missing values")
    print(f"Original shape: {df.shape}, Cleaned shape: {df_clean.shape}")
    
    return df_clean

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        Boolean indicating if dataset is valid
    """
    if df.empty:
        print("Error: Dataset is empty")
        return False
    
    if df.shape[0] < min_rows:
        print(f"Error: Dataset has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve', 'Eve'],
        'age': [25, 30, 30, None, 35, 40, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'], fill_missing='mean')
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    is_valid = validate_dataset(cleaned_df, required_columns=['id', 'name', 'age', 'score'], min_rows=1)
    print(f"\nDataset validation: {'PASS' if is_valid else 'FAIL'}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
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
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
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
        'count': df[column].count()
    }
    
    return stats

def process_dataframe(df, numeric_columns):
    """
    Process multiple numeric columns to remove outliers and return cleaned DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Cleaned DataFrame with outliers removed from specified columns
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("Original summary for column 'A':")
    print(calculate_summary_statistics(df, 'A'))
    
    cleaned_df = process_dataframe(df, ['A', 'B', 'C'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary for column 'A':")
    print(calculate_summary_statistics(cleaned_df, 'A'))