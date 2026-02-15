import pandas as pd
import numpy as np
import re

def clean_csv_data(input_file, output_file):
    """
    Clean a CSV file by removing duplicates, handling missing values,
    standardizing text columns, and converting data types.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(df)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        text_cols = df.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in text_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna('Unknown', inplace=True)
        
        # Standardize text columns (lowercase, strip whitespace)
        for col in text_cols:
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
        
        # Convert date columns if present
        date_pattern = r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}'
        for col in df.columns:
            if df[col].astype(str).str.match(date_pattern).any():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        
        # Generate cleaning report
        report = {
            'input_file': input_file,
            'output_file': output_file,
            'original_rows': initial_count,
            'final_rows': len(df),
            'duplicates_removed': duplicates_removed,
            'numeric_columns': len(numeric_cols),
            'text_columns': len(text_cols),
            'missing_values_filled': df.isnull().sum().sum()
        }
        
        return report
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_cleaned_data(file_path):
    """
    Validate the cleaned data file for basic integrity checks.
    """
    try:
        df = pd.read_csv(file_path)
        
        validation_results = {
            'file_exists': True,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'has_duplicates': len(df) != len(df.drop_duplicates()),
            'has_null_values': df.isnull().sum().sum() > 0,
            'column_names': list(df.columns),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        return validation_results
        
    except Exception as e:
        return {'file_exists': False, 'error': str(e)}

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    print(f"Starting data cleaning process...")
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    
    cleaning_report = clean_csv_data(input_csv, output_csv)
    
    if cleaning_report:
        print("\nCleaning Report:")
        for key, value in cleaning_report.items():
            print(f"{key}: {value}")
        
        print("\nValidating cleaned data...")
        validation = validate_cleaned_data(output_csv)
        
        if validation['file_exists']:
            print("\nValidation Results:")
            for key, value in validation.items():
                if key not in ['column_names', 'data_types']:
                    print(f"{key}: {value}")
        else:
            print(f"Validation failed: {validation.get('error', 'Unknown error')}")
    else:
        print("Data cleaning failed.")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str): Method to fill missing values. 
                            Options: 'mean', 'median', 'mode', or 'drop'. 
                            Default is 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            elif fill_missing == 'mode':
                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_na=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fill_na (str or None): Method to fill missing values. Options: 
                           'mean', 'median', 'mode', or None to drop rows.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_na is None:
        cleaned_df = cleaned_df.dropna()
    elif fill_na == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_na == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fill_na == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0)
    
    # Remove duplicates
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (bool, str) indicating success and message.
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to analyze
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        Boolean mask of outliers
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].fillna(0)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].fillna(0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values with specified strategy.
    
    Args:
        data: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    data_clean = data.copy()
    
    for col in columns:
        if strategy == 'mean':
            fill_value = data[col].mean()
        elif strategy == 'median':
            fill_value = data[col].median()
        elif strategy == 'mode':
            fill_value = data[col].mode()[0] if not data[col].mode().empty else 0
        elif strategy == 'drop':
            data_clean = data_clean.dropna(subset=[col])
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        data_clean[col] = data[col].fillna(fill_value)
    
    return data_clean

def process_dataframe(df, config):
    """
    Main function to process DataFrame with given configuration.
    
    Args:
        df: Input DataFrame
        config: Dictionary with processing configuration
    
    Returns:
        Processed DataFrame
    """
    result_df = df.copy()
    
    # Handle missing values
    if 'missing_values' in config:
        strategy = config['missing_values'].get('strategy', 'mean')
        columns = config['missing_values'].get('columns')
        result_df = clean_missing_values(result_df, strategy, columns)
    
    # Remove outliers
    if 'remove_outliers' in config:
        for col in config['remove_outliers'].get('columns', []):
            if col in result_df.columns:
                result_df = remove_outliers_zscore(
                    result_df, 
                    col, 
                    config['remove_outliers'].get('threshold', 3)
                )
    
    # Normalize columns
    if 'normalize' in config:
        method = config['normalize'].get('method', 'minmax')
        for col in config['normalize'].get('columns', []):
            if col in result_df.columns:
                if method == 'minmax':
                    result_df[col] = normalize_minmax(result_df, col)
                elif method == 'zscore':
                    result_df[col] = normalize_zscore(result_df, col)
    
    return result_df