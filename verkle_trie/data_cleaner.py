def remove_duplicates(input_list):
    """
    Remove duplicate items from a list while preserving the original order.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_strings(string_list):
    """
    Clean a list of strings by converting numeric strings to integers.
    Non-numeric strings are kept as-is.
    """
    cleaned = []
    for s in string_list:
        s_stripped = s.strip()
        if s_stripped.isdigit():
            cleaned.append(int(s_stripped))
        else:
            cleaned.append(s)
    return cleaned

def filter_by_type(data_list, data_type):
    """
    Filter a list to include only items of a specific type.
    """
    return [item for item in data_list if isinstance(item, data_type)]

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, "4", "4", "five", 5.0, 5.0]
    print("Original:", sample_data)
    
    unique_data = remove_duplicates(sample_data)
    print("After removing duplicates:", unique_data)
    
    cleaned_data = clean_numeric_strings(unique_data)
    print("After cleaning numeric strings:", cleaned_data)
    
    integers_only = filter_by_type(cleaned_data, int)
    print("Integers only:", integers_only)import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, columns=None):
    """
    Normalize data using min-max scaling
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val > min_val:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
    
    return normalized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    cleaned_data = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if data[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(data[col]):
                cleaned_data[col].fillna(data[col].mean(), inplace=True)
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(data[col]):
                cleaned_data[col].fillna(data[col].median(), inplace=True)
            elif strategy == 'mode':
                cleaned_data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 0, inplace=True)
            elif strategy == 'drop':
                cleaned_data = cleaned_data.dropna(subset=[col])
    
    return cleaned_data

def clean_dataset(data, config):
    """
    Main cleaning pipeline
    """
    cleaned_data = data.copy()
    
    if 'outlier_columns' in config:
        for col in config['outlier_columns']:
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    if 'missing_strategy' in config:
        cleaned_data = handle_missing_values(
            cleaned_data, 
            strategy=config['missing_strategy'],
            columns=config.get('missing_columns')
        )
    
    if 'normalize_columns' in config:
        cleaned_data = normalize_minmax(cleaned_data, config['normalize_columns'])
    
    return cleaned_dataimport pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: If True, fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        if removed > 0:
            print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        missing_after = cleaned_df.isnull().sum().sum()
        if missing_before > 0:
            print(f"Filled {missing_before} missing values")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Boolean indicating if validation passed
    """
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def get_dataframe_stats(df):
    """
    Get basic statistics about a DataFrame.
    
    Args:
        df: pandas DataFrame to analyze
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    }
    
    return statsimport pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, overwrites input file
        subset (list, optional): List of column names to consider for identifying duplicates
    """
    try:
        df = pd.read_csv(input_file)
        
        if subset:
            df_clean = df.drop_duplicates(subset=subset)
        else:
            df_clean = df.drop_duplicates()
        
        removed_count = len(df) - len(df_clean)
        
        if output_file is None:
            output_file = input_file
        
        df_clean.to_csv(output_file, index=False)
        
        print(f"Removed {removed_count} duplicate rows")
        print(f"Original rows: {len(df)}")
        print(f"Cleaned rows: {len(df_clean)}")
        print(f"Saved to: {output_file}")
        
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
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

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 13, 12, 11, 10, 14, 13, 12, 11, 10]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    print(f"\nOriginal stats: {calculate_basic_stats(df, 'values')}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned data:")
    print(cleaned_df)
    print(f"\nCleaned stats: {calculate_basic_stats(cleaned_df, 'values')}")