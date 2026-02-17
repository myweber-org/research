
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    mean = df[column].mean()
    std = df[column].std()
    df[column] = (df[column] - mean) / std
    return df

def clean_dataset(input_file, output_file):
    df = load_data(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        data: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    for col in columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            col_min = normalized_data[col].min()
            col_max = normalized_data[col].max()
            if col_max != col_min:
                normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
            else:
                normalized_data[col] = 0
    
    return normalized_data

def standardize_zscore(data, columns=None):
    """
    Standardize specified columns using z-score normalization.
    
    Args:
        data: pandas DataFrame
        columns: list of column names to standardize (default: all numeric columns)
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    standardized_data = data.copy()
    for col in columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            col_mean = standardized_data[col].mean()
            col_std = standardized_data[col].std()
            if col_std > 0:
                standardized_data[col] = (standardized_data[col] - col_mean) / col_std
            else:
                standardized_data[col] = 0
    
    return standardized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', or 'drop')
        columns: list of column names to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = data.columns
    
    processed_data = data.copy()
    
    for col in columns:
        if col not in processed_data.columns:
            continue
            
        if processed_data[col].isnull().any():
            if strategy == 'drop':
                processed_data = processed_data.dropna(subset=[col])
            elif strategy == 'mean' and np.issubdtype(processed_data[col].dtype, np.number):
                processed_data[col].fillna(processed_data[col].mean(), inplace=True)
            elif strategy == 'median' and np.issubdtype(processed_data[col].dtype, np.number):
                processed_data[col].fillna(processed_data[col].median(), inplace=True)
            elif strategy == 'mode':
                processed_data[col].fillna(processed_data[col].mode()[0], inplace=True)
    
    return processed_data

def clean_dataset(data, config):
    """
    Apply multiple cleaning operations based on configuration.
    
    Args:
        data: pandas DataFrame
        config: dictionary with cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if 'missing_values' in config:
        cleaned_data = handle_missing_values(
            cleaned_data,
            strategy=config['missing_values'].get('strategy', 'mean'),
            columns=config['missing_values'].get('columns')
        )
    
    if 'outliers' in config:
        for col_config in config['outliers']:
            cleaned_data = remove_outliers_iqr(
                cleaned_data,
                column=col_config['column'],
                multiplier=col_config.get('multiplier', 1.5)
            )
    
    if 'normalization' in config:
        norm_type = config['normalization'].get('type', 'minmax')
        columns = config['normalization'].get('columns')
        
        if norm_type == 'minmax':
            cleaned_data = normalize_minmax(cleaned_data, columns)
        elif norm_type == 'zscore':
            cleaned_data = standardize_zscore(cleaned_data, columns)
    
    return cleaned_data
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
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

def get_cleaning_stats(original_df, cleaned_df):
    """
    Get statistics about the data cleaning process.
    
    Args:
        original_df (pd.DataFrame): Original DataFrame
        cleaned_df (pd.DataFrame): Cleaned DataFrame
    
    Returns:
        dict: Cleaning statistics
    """
    stats = {
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'rows_removed': len(original_df) - len(cleaned_df),
        'removal_percentage': round(((len(original_df) - len(cleaned_df)) / len(original_df)) * 100, 2)
    }
    
    return stats

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 200],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = clean_numeric_data(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print()
    
    stats = get_cleaning_stats(df, cleaned_df)
    print("Cleaning Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
import numpy as np
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
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
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
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='zscore'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    report = {
        'original_rows': len(data),
        'outliers_removed': 0,
        'columns_normalized': []
    }
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data, removed = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data, removed = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
        
        report['outliers_removed'] += removed
        
        if normalize_method == 'minmax':
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
        
        report['columns_normalized'].append(column)
    
    report['final_rows'] = len(cleaned_data)
    report['rows_removed'] = report['original_rows'] - report['final_rows']
    
    return cleaned_data, report
import numpy as np
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

def zscore_normalization(data, column):
    """
    Normalize data using z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column]
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def minmax_normalization(data, column, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column]
    
    a, b = feature_range
    normalized = a + ((data[column] - min_val) * (b - a)) / (max_val - min_val)
    return normalized

def detect_missing_patterns(data, threshold=0.3):
    """
    Detect columns with high percentage of missing values
    """
    missing_percentage = data.isnull().sum() / len(data) * 100
    high_missing_cols = missing_percentage[missing_percentage > threshold].index.tolist()
    
    missing_info = {
        'total_missing': data.isnull().sum().sum(),
        'missing_percentage': missing_percentage,
        'high_missing_columns': high_missing_cols
    }
    
    return missing_info

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5, normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    cleaning_report = {
        'original_shape': data.shape,
        'outliers_removed': {},
        'normalized_columns': []
    }
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            try:
                filtered_data, outliers = remove_outliers_iqr(cleaned_data, col, outlier_factor)
                cleaning_report['outliers_removed'][col] = outliers
                cleaned_data = filtered_data
                
                if normalize_method == 'zscore':
                    cleaned_data[f'{col}_normalized'] = zscore_normalization(cleaned_data, col)
                elif normalize_method == 'minmax':
                    cleaned_data[f'{col}_normalized'] = minmax_normalization(cleaned_data, col)
                
                cleaning_report['normalized_columns'].append(col)
            except Exception as e:
                print(f"Error processing column {col}: {str(e)}")
                continue
    
    cleaning_report['final_shape'] = cleaned_data.shape
    cleaning_report['rows_removed'] = data.shape[0] - cleaned_data.shape[0]
    cleaning_report['columns_added'] = len(cleaned_data.columns) - len(data.columns)
    
    return cleaned_data, cleaning_report

def validate_cleaned_data(data, original_data):
    """
    Validate that cleaning didn't corrupt the data
    """
    validation_results = {
        'has_nulls': data.isnull().sum().sum() == 0,
        'shape_changed': data.shape != original_data.shape,
        'columns_preserved': all(col in data.columns for col in original_data.columns),
        'dtypes_consistent': all(data[col].dtype == original_data[col].dtype 
                                for col in original_data.columns if col in data.columns)
    }
    
    return validation_results
import pandas as pd

def clean_dataframe(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling null values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values. Default True.
    column_case (str): Target case for column names ('lower', 'upper', 'title'). Default 'lower'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Handle null values
    if drop_na:
        df_clean = df_clean.dropna()
    else:
        df_clean = df_clean.fillna(0)
    
    # Standardize column names
    if column_case == 'lower':
        df_clean.columns = df_clean.columns.str.lower()
    elif column_case == 'upper':
        df_clean.columns = df_clean.columns.str.upper()
    elif column_case == 'title':
        df_clean.columns = df_clean.columns.str.title()
    
    # Remove any leading/trailing whitespace from column names
    df_clean.columns = df_clean.columns.str.strip()
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, None, 35, 40],
        'Score': [85.5, 92.0, 78.5, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataframe(df, drop_na=True, column_case='lower')
    print(cleaned_df)
    
    # Validation test
    is_valid = validate_dataframe(cleaned_df, required_columns=['name', 'age'])
    print(f"\nDataFrame validation: {is_valid}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
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
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    removal_stats = {}
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data, removed = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data, removed = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError("Invalid outlier method. Use 'iqr' or 'zscore'")
        
        removal_stats[column] = removed
        
        if normalize_method == 'minmax':
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError("Invalid normalize method. Use 'minmax' or 'zscore'")
    
    return cleaned_data, removal_stats

def validate_data(data, required_columns, numeric_columns):
    """
    Validate data structure and content
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    for column in numeric_columns:
        if column in data.columns:
            if not pd.api.types.is_numeric_dtype(data[column]):
                raise ValueError(f"Column '{column}' must be numeric")
    
    return True

def generate_summary(data, numeric_columns):
    """
    Generate statistical summary of the data
    """
    summary = {}
    
    for column in numeric_columns:
        if column in data.columns:
            summary[column] = {
                'mean': data[column].mean(),
                'std': data[column].std(),
                'min': data[column].min(),
                'max': data[column].max(),
                'median': data[column].median(),
                'count': data[column].count(),
                'missing': data[column].isnull().sum()
            }
    
    return summaryimport pandas as pd

def clean_dataset(df, columns_to_check=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list, optional): List of column names to check for duplicates.
                                       If None, checks all columns.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str or value): Method to fill missing values.
                                 Can be 'mean', 'median', 'mode', or a scalar value.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        if columns_to_check:
            cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
        else:
            cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        for column in cleaned_df.select_dtypes(include=['number']).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
                else:
                    cleaned_df[column].fillna(fill_missing, inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df.empty:
        print("DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, 20.3, 20.3, None, 40.1],
        'category': ['A', 'B', 'B', 'C', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df, columns_to_check=['id', 'category'], fill_missing='median')
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['id', 'value'], min_rows=3)
    print(f"\nData validation passed: {is_valid}")