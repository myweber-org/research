
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values
        strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_rows = initial_rows - len(cleaned_df)
        print(f"Removed {removed_rows} duplicate rows")
    
    if fill_missing and cleaned_df.isnull().sum().any():
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().sum() > 0:
                if strategy == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif strategy == 'median':
                    fill_value = cleaned_df[col].median()
                elif strategy == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {strategy}: {fill_value}")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().sum() > 0:
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
            print(f"Filled missing values in categorical column '{col}' with 'Unknown'")
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if len(df) < min_rows:
        print(f"Validation failed: Dataset has only {len(df)} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Validation failed: Missing required columns: {missing_columns}")
            return False
    
    if df.isnull().sum().sum() > 0:
        print(f"Warning: Dataset contains {df.isnull().sum().sum()} missing values")
    
    return True

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path to save the file
        format (str): File format ('csv', 'parquet', 'json')
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    is_valid = validate_dataset(cleaned_df, required_columns=['id', 'name', 'age', 'score'], min_rows=1)
    print(f"\nDataset validation: {'PASSED' if is_valid else 'FAILED'}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_na (bool): Whether to drop rows with null values
    rename_columns (bool): Whether to standardize column names
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    if drop_na:
        df_clean = df_clean.dropna()
    
    if rename_columns:
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('DataFrame is empty')
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing columns: {missing_cols}')
    
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        validation_results['warnings'].append(f'Found {null_count} null values in DataFrame')
    
    return validation_results

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list): Columns to consider for duplicates
    keep (str): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_numeric_columns(df, columns=None):
    """
    Normalize numeric columns to range [0, 1].
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Specific columns to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    else:
        numeric_cols = [col for col in columns if col in df.columns]
    
    for col in numeric_cols:
        col_min = df_normalized[col].min()
        col_max = df_normalized[col].max()
        
        if col_max > col_min:
            df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
    
    return df_normalized
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if normalized_df[col].dtype not in [np.number, 'int64', 'float64']:
            continue
        
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        
        if col_max - col_min == 0:
            normalized_df[col] = 0
        else:
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
    
    return normalized_df

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Detect columns with skewed distributions.
    
    Args:
        dataframe: pandas DataFrame
        threshold: Absolute skewness threshold for detection
    
    Returns:
        Dictionary with column names and their skewness values
    """
    skewed_columns = {}
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        skewness = stats.skew(dataframe[col].dropna())
        if abs(skewness) > threshold:
            skewed_columns[col] = skewness
    
    return skewed_columns

def clean_dataset(dataframe, outlier_columns=None, normalize=True, skew_threshold=1.0):
    """
    Comprehensive dataset cleaning pipeline.
    
    Args:
        dataframe: Input DataFrame
        outlier_columns: Columns to apply outlier removal. If None, use all numeric columns.
        normalize: Whether to normalize numeric columns
        skew_threshold: Skewness threshold for reporting
    
    Returns:
        Tuple of (cleaned DataFrame, cleaning report)
    """
    cleaned_df = dataframe.copy()
    report = {
        'original_shape': dataframe.shape,
        'outliers_removed': 0,
        'skewed_columns': {},
        'normalized_columns': []
    }
    
    if outlier_columns is None:
        outlier_columns = list(dataframe.select_dtypes(include=[np.number]).columns)
    
    for col in outlier_columns:
        if col in cleaned_df.columns and cleaned_df[col].dtype in [np.number, 'int64', 'float64']:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            report['outliers_removed'] += original_count - len(cleaned_df)
    
    if normalize:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df = normalize_minmax(cleaned_df, list(numeric_cols))
        report['normalized_columns'] = list(numeric_cols)
    
    report['skewed_columns'] = detect_skewed_columns(cleaned_df, skew_threshold)
    report['final_shape'] = cleaned_df.shape
    
    return cleaned_df, report

def save_cleaning_report(report, filepath):
    """
    Save cleaning report to a text file.
    
    Args:
        report: Cleaning report dictionary
        filepath: Path to save the report
    """
    with open(filepath, 'w') as f:
        f.write("Data Cleaning Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Original dataset shape: {report['original_shape']}\n")
        f.write(f"Final dataset shape: {report['final_shape']}\n")
        f.write(f"Total outliers removed: {report['outliers_removed']}\n\n")
        
        f.write("Normalized columns:\n")
        for col in report['normalized_columns']:
            f.write(f"  - {col}\n")
        
        f.write("\nSkewed columns (|skewness| > 1.0):\n")
        for col, skew in report['skewed_columns'].items():
            f.write(f"  - {col}: {skew:.3f}\n")