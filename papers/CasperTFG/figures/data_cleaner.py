
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from specified numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_columns (list): List of numeric column names to clean.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.randn(1000) * 10,
        'B': np.random.randn(1000) * 5 + 20,
        'C': np.random.randn(1000) * 2 + 100
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original dataset shape: {df.shape}")
    
    cleaned_df = clean_dataset(df)
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
def remove_duplicates(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
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
    
    return filtered_df.reset_index(drop=True)

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean. If None, uses all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.randint(1, 100, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[10, 'A'] = 500
    df.loc[20, 'B'] = 1000
    
    print("Original DataFrame shape:", df.shape)
    print("\nData validation results:")
    validation = validate_dataframe(df)
    for key, value in validation.items():
        print(f"{key}: {value}")
    
    cleaned_df = clean_numeric_data(df, ['A', 'B'])
    print("\nCleaned DataFrame shape:", cleaned_df.shape)import pandas as pd
import numpy as np

def clean_data(input_file, output_file):
    """
    Load a CSV file, perform basic cleaning operations,
    and save the cleaned data to a new file.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")

        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")

        df = df.dropna(subset=['critical_column'])
        print(f"After dropping rows with missing critical_column: {df.shape}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        return True

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except KeyError as e:
        print(f"Error: Required column not found - {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    clean_data('raw_data.csv', 'cleaned_data.csv')
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fillna_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fillna_method (str): Method to handle missing values ('drop', 'mean', 'median', 'mode', 'zero').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fillna_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fillna_method == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fillna_method == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fillna_method == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif fillna_method == 'zero':
        cleaned_df = cleaned_df.fillna(0)
    else:
        raise ValueError(f"Unsupported fillna_method: {fillna_method}")
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (np.ndarray): Input data array
    column (int): Column index to process
    
    Returns:
    np.ndarray: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (np.ndarray): Input data array
    column (int): Column index to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    col_data = data[:, column]
    
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data),
        'min': np.min(col_data),
        'max': np.max(col_data),
        'q1': np.percentile(col_data, 25),
        'q3': np.percentile(col_data, 75)
    }
    
    return stats

def process_dataset(data, target_columns):
    """
    Process multiple columns for outlier removal and statistics.
    
    Parameters:
    data (np.ndarray): Input data array
    target_columns (list): List of column indices to process
    
    Returns:
    tuple: (cleaned_data, statistics_dict)
    """
    cleaned_data = data.copy()
    all_stats = {}
    
    for col in target_columns:
        cleaned_data = remove_outliers_iqr(cleaned_data, col)
        all_stats[col] = calculate_statistics(cleaned_data, col)
    
    return cleaned_data, all_stats

def validate_data_shape(data, expected_columns):
    """
    Validate that data has the expected number of columns.
    
    Parameters:
    data (np.ndarray): Input data array
    expected_columns (int): Expected number of columns
    
    Returns:
    bool: True if shape is valid, False otherwise
    """
    if data.shape[1] != expected_columns:
        print(f"Warning: Expected {expected_columns} columns, got {data.shape[1]}")
        return False
    return True

def example_usage():
    """
    Example demonstrating the usage of data cleaning functions.
    """
    np.random.seed(42)
    
    sample_data = np.random.randn(100, 3)
    sample_data[:, 1] = sample_data[:, 1] * 5 + 10
    
    print("Original data shape:", sample_data.shape)
    
    target_cols = [0, 1]
    
    if validate_data_shape(sample_data, 3):
        cleaned_data, statistics = process_dataset(sample_data, target_cols)
        
        print("Cleaned data shape:", cleaned_data.shape)
        print("Statistics for column 1:", statistics[1])

if __name__ == "__main__":
    example_usage()
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
            cleaned_df = standardize_zscore(cleaned_df, col)
    return cleaned_df

def load_and_clean_data(filepath):
    try:
        df = pd.read_csv(filepath)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return clean_dataset(df, numeric_cols)
    except Exception as e:
        print(f"Error processing file: {e}")
        return None
import pandas as pd

def clean_dataset(df, columns_to_check=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): List of column names to check for duplicates.
            If None, checks all columns. Defaults to None.
        drop_duplicates (bool, optional): Whether to drop duplicate rows. Defaults to True.
        fill_missing (str, optional): Method to fill missing values.
            Options: 'mean', 'median', 'mode', or 'drop'. Defaults to 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates(subset=columns_to_check, keep='first')
    
    if fill_missing == 'drop':
        df_clean = df_clean.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            elif fill_missing == 'median':
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif fill_missing == 'mode':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False
    
    return True
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str or dict): Method to fill missing values. 
                                Can be 'mean', 'median', 'mode', or a dictionary of column:value pairs.
                                If None, missing values are not filled. Default is None.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df

def validate_data(df, required_columns=None, unique_columns=None):
    """
    Validate the DataFrame for required columns and unique constraints.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    unique_columns (list): List of column names that should have unique values.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'duplicate_values': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    if unique_columns:
        for col in unique_columns:
            if col in df.columns:
                duplicates = df[df.duplicated(subset=[col], keep=False)]
                if not duplicates.empty:
                    validation_results['duplicate_values'][col] = len(duplicates)
                    validation_results['is_valid'] = False
    
    return validation_results
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(filepath):
    """
    Load CSV data and perform cleaning operations.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    # Remove duplicates
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    removed = initial_rows - df.shape[0]
    print(f"Removed {removed} duplicate rows")
    
    # Normalize numeric columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        print(f"Normalized {len(numeric_cols)} numeric columns")
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    print(f"Final data shape: {df.shape}")
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return True
    return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    if cleaned_df is not None:
        save_cleaned_data(cleaned_df, output_file)
def remove_duplicates_preserve_order(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import re

def clean_dataframe(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing
    specified string columns (strip whitespace, lowercase).
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize specified string columns
    for col in column_names:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.lower()
    
    return df_cleaned

def remove_special_characters(text):
    """
    Remove special characters from a string, keeping only alphanumeric and spaces.
    """
    if pd.isna(text):
        return text
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(text))

def validate_email(email):
    """
    Validate email format using regex pattern.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if pd.isna(email):
        return False
    return bool(re.match(pattern, str(email)))