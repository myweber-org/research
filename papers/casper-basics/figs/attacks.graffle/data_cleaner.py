import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    file_path (str): Path to input CSV file
    output_path (str): Path for cleaned output CSV (optional)
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        original_rows = df.shape[0]
        
        df = df.drop_duplicates()
        print(f"Removed {original_rows - df.shape[0]} duplicate rows")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mean(), inplace=True)
        elif missing_strategy == 'median':
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
        elif missing_strategy == 'drop':
            df = df.dropna(subset=numeric_cols)
        
        print(f"Missing values handled using '{missing_strategy}' strategy")
        print(f"Final data shape: {df.shape}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    if df.isnull().any().any():
        print("Warning: DataFrame contains missing values")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'category': ['A', 'B', 'A', 'C', 'B', 'A']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv', 'mean')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, ['id', 'value', 'category'])
        print(f"Data validation result: {is_valid}")
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if normalize_text and columns_to_clean:
        for col in columns_to_clean:
            if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
                print(f"Normalized text in column: {col}")
    
    return cleaned_df

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].apply(lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False)
    
    valid_count = df['email_valid'].sum()
    total_count = df[email_column].notna().sum()
    
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.2f}%)")
    
    return df
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric columns by removing outliers and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'values': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10, 200],
        'other_values': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_numeric_data(df, columns=['values'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def normalize_minmax(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    if max_val == min_val:
        return dataframe[column].apply(lambda x: 0.5)
    return dataframe[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def standardize_zscore(dataframe, column):
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    if std_val == 0:
        return dataframe[column].apply(lambda x: 0)
    return dataframe[column].apply(lambda x: (x - mean_val) / std_val)

def clean_dataset(dataframe, numeric_columns):
    cleaned_df = dataframe.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col + '_normalized'] = normalize_minmax(cleaned_df, col)
            cleaned_df[col + '_standardized'] = standardize_zscore(cleaned_df, col)
    return cleaned_df

def validate_cleaning(dataframe, original_dataframe, column):
    if column not in dataframe.columns or column not in original_dataframe.columns:
        return False
    original_std = original_dataframe[column].std()
    cleaned_std = dataframe[column].std()
    return cleaned_std <= original_stdimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].copy()
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].copy()
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    stats_report = {}
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        original_count = len(cleaned_df)
        
        if outlier_method == 'iqr':
            cleaned_df, outliers = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df, outliers = remove_outliers_zscore(cleaned_df, col)
        else:
            outliers = 0
        
        if normalize_method == 'minmax':
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[f'{col}_standardized'] = normalize_zscore(cleaned_df, col)
        
        stats_report[col] = {
            'original_samples': original_count,
            'outliers_removed': outliers,
            'remaining_samples': len(cleaned_df)
        }
    
    return cleaned_df, stats_report

def validate_data(df, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan:
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Dataset contains {nan_count} NaN values")
    
    return True