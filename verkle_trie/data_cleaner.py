import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Strategy to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}.")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
        print("Filled missing categorical values with mode.")
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_cols
        validation_results['all_required_columns_present'] = len(missing_cols) == 0
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_data(df))
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nCleaned Validation Results:")
    print(validate_data(cleaned))
import pandas as pd
import numpy as np

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

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def validate_numeric_data(df, columns):
    """
    Validate that specified columns contain only numeric data.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to validate
    
    Returns:
    dict: Dictionary with validation results for each column
    """
    results = {}
    
    for col in columns:
        if col not in df.columns:
            results[col] = {'valid': False, 'error': 'Column not found'}
            continue
        
        non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
        total_count = df[col].count()
        
        results[col] = {
            'valid': non_numeric == 0,
            'non_numeric_count': non_numeric,
            'total_count': total_count,
            'percentage_valid': ((total_count - non_numeric) / total_count * 100) if total_count > 0 else 0
        }
    
    return results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'values': [1, 2, 3, 4, 5, 100, 200, 300, 400, 500],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("DataFrame after removing outliers:")
    print(cleaned_df)
    print()
    
    stats = calculate_summary_stats(df, 'values')
    print("Summary statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()
    
    validation = validate_numeric_data(df, ['values', 'category'])
    print("Data validation results:")
    for col, result in validation.items():
        print(f"{col}: {result}")
import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, fill_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    file_path (str): Path to input CSV file.
    output_path (str, optional): Path for cleaned output CSV.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero').
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    
    df = pd.read_csv(file_path)
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    df = df.drop_duplicates()
    print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    missing_before = df.isnull().sum().sum()
    
    if fill_strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif fill_strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif fill_strategy == 'mode':
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    elif fill_strategy == 'zero':
        df = df.fillna(0)
    
    missing_after = df.isnull().sum().sum()
    print(f"Filled {missing_before - missing_after} missing values")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    print(f"Final data shape: {df.shape}")
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of required column names.
    
    Returns:
    dict: Validation results.
    """
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
    
    if df.isnull().sum().sum() > 0:
        validation_results['warnings'].append("DataFrame contains missing values")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].std() == 0:
            validation_results['warnings'].append(f"Column '{col}' has zero variance")
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, None, 12.8, 12.8],
        'category': ['A', 'B', None, 'A', 'C', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', 'cleaned_sample.csv', 'mean')
    
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
    print(f"Validation results: {validation}")
import csv
import os

def clean_csv(input_path, output_path, delimiter=',', quotechar='"'):
    """
    Clean a CSV file by removing rows with missing values and trimming whitespace.
    
    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the cleaned CSV file.
        delimiter (str): CSV delimiter character.
        quotechar (str): CSV quote character.
    
    Returns:
        int: Number of rows in the cleaned file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    cleaned_rows = []
    
    with open(input_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=delimiter, quotechar=quotechar)
        
        for row in reader:
            # Skip rows with any empty or whitespace-only cells
            if any(cell.strip() == '' for cell in row):
                continue
            
            # Trim whitespace from each cell
            cleaned_row = [cell.strip() for cell in row]
            cleaned_rows.append(cleaned_row)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter=delimiter, quotechar=quotechar, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(cleaned_rows)
    
    return len(cleaned_rows)

def validate_csv_headers(file_path, required_headers, delimiter=','):
    """
    Validate that a CSV file contains all required headers.
    
    Args:
        file_path (str): Path to the CSV file.
        required_headers (list): List of required header names.
        delimiter (str): CSV delimiter character.
    
    Returns:
        bool: True if all required headers are present.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        headers = next(reader, [])
    
    headers = [header.strip().lower() for header in headers]
    required_lower = [header.strip().lower() for header in required_headers]
    
    missing_headers = [header for header in required_lower if header not in headers]
    
    if missing_headers:
        print(f"Missing headers: {missing_headers}")
        return False
    
    return True

def get_csv_stats(file_path, delimiter=','):
    """
    Get basic statistics about a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        delimiter (str): CSV delimiter character.
    
    Returns:
        dict: Dictionary containing row count, column count, and sample data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        rows = list(reader)
    
    if not rows:
        return {
            'row_count': 0,
            'column_count': 0,
            'headers': [],
            'sample_rows': []
        }
    
    row_count = len(rows) - 1  # Exclude header
    column_count = len(rows[0]) if rows else 0
    headers = rows[0] if rows else []
    sample_rows = rows[1:6] if len(rows) > 1 else []  # First 5 data rows
    
    return {
        'row_count': row_count,
        'column_count': column_count,
        'headers': headers,
        'sample_rows': sample_rows
    }

if __name__ == "__main__":
    # Example usage
    try:
        stats = get_csv_stats('sample_data.csv')
        print(f"CSV Statistics: {stats}")
        
        if validate_csv_headers('sample_data.csv', ['id', 'name', 'email']):
            cleaned_count = clean_csv('sample_data.csv', 'cleaned_data.csv')
            print(f"Cleaned {cleaned_count} rows")
        else:
            print("CSV validation failed")
    except Exception as e:
        print(f"Error: {e}")
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if dataframe.empty:
        return dataframe
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    
    return dataframe

def validate_dataframe(dataframe, required_columns):
    """
    Validate that DataFrame contains required columns.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if all required columns are present
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    return True