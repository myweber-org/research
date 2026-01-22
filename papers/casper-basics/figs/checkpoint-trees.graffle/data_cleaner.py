
import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values with specified strategy
    """
    if columns is None:
        columns = data.columns
    
    data_clean = data.copy()
    
    for col in columns:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_clean[col].fillna(fill_value, inplace=True)
    
    return data_clean

def remove_duplicates(data, subset=None, keep='first'):
    """
    Remove duplicate rows from dataset
    """
    return data.drop_duplicates(subset=subset, keep=keep)

def normalize_data(data, columns=None, method='minmax'):
    """
    Normalize data using specified method
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    data_normalized = data.copy()
    
    for col in columns:
        if method == 'minmax':
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val != min_val:
                data_normalized[col] = (data[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val != 0:
                data_normalized[col] = (data[col] - mean_val) / std_val
        
        elif method == 'robust':
            median_val = data[col].median()
            iqr_val = stats.iqr(data[col])
            if iqr_val != 0:
                data_normalized[col] = (data[col] - median_val) / iqr_val
    
    return data_normalized

def clean_dataset(data, missing_strategy='mean', normalize_method=None, remove_outliers=False):
    """
    Comprehensive data cleaning pipeline
    """
    # Handle missing values
    cleaned_data = handle_missing_values(data, strategy=missing_strategy)
    
    # Remove duplicates
    cleaned_data = remove_duplicates(cleaned_data)
    
    # Remove outliers if requested
    if remove_outliers:
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outliers = detect_outliers_iqr(cleaned_data, col)
            cleaned_data = cleaned_data.drop(outliers.index)
    
    # Normalize if requested
    if normalize_method:
        cleaned_data = normalize_data(cleaned_data, method=normalize_method)
    
    return cleaned_data
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    df = pd.read_csv(input_file)
    
    df.replace('', np.nan, inplace=True)
    df.dropna(subset=['critical_column'], inplace=True)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_csv_data('raw_data.csv', 'cleaned_data.csv')
    print(f"Data cleaning complete. Shape: {cleaned_df.shape}")
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=np.nan):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_value: Value to use for filling missing data.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing:
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        missing_after = cleaned_df.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values.")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("Warning: DataFrame is empty.")
    
    return True

def main():
    """
    Example usage of the data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
        'age': [25, 30, 30, None, 28, 28],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    if validate_dataframe(df, required_columns=['id', 'name']):
        cleaned_df = clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0)
        print("\nCleaned DataFrame:")
        print(cleaned_df)
        
        print("\nCleaned DataFrame info:")
        print(cleaned_df.info())
    else:
        print("Data validation failed.")

if __name__ == "__main__":
    main()