
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def zscore_normalize(dataframe, column):
    """
    Normalize a column using z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    
    if std_val == 0:
        return dataframe[column] - mean_val
    
    normalized = (dataframe[column] - mean_val) / std_val
    return normalized

def minmax_normalize(dataframe, column, feature_range=(0, 1)):
    """
    Normalize a column using min-max scaling.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to normalize
        feature_range: Desired range of transformed data (default 0-1)
    
    Returns:
        Series with normalized values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if min_val == max_val:
        return dataframe[column] * 0 + feature_range[0]
    
    normalized = (dataframe[column] - min_val) / (max_val - min_val)
    normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return normalized

def detect_missing_patterns(dataframe, threshold=0.3):
    """
    Detect columns with high percentage of missing values.
    
    Args:
        dataframe: pandas DataFrame
        threshold: Missing value percentage threshold (default 0.3)
    
    Returns:
        List of column names exceeding the threshold
    """
    missing_ratios = dataframe.isnull().sum() / len(dataframe)
    high_missing_cols = missing_ratios[missing_ratios > threshold].index.tolist()
    
    return high_missing_cols

def clean_dataset(dataframe, outlier_columns=None, normalize_columns=None, 
                  normalization_method='zscore', missing_threshold=0.3):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: Input DataFrame
        outlier_columns: List of columns to remove outliers from
        normalize_columns: List of columns to normalize
        normalization_method: 'zscore' or 'minmax'
        missing_threshold: Threshold for identifying high-missing columns
    
    Returns:
        Cleaned DataFrame and cleaning report
    """
    df_clean = dataframe.copy()
    report = {}
    
    # Handle missing values
    high_missing = detect_missing_patterns(df_clean, missing_threshold)
    report['high_missing_columns'] = high_missing
    
    if high_missing:
        df_clean = df_clean.drop(columns=high_missing)
    
    # Remove outliers
    if outlier_columns:
        initial_rows = len(df_clean)
        for col in outlier_columns:
            if col in df_clean.columns:
                df_clean = remove_outliers_iqr(df_clean, col)
        report['outliers_removed'] = initial_rows - len(df_clean)
    
    # Normalize columns
    if normalize_columns:
        for col in normalize_columns:
            if col in df_clean.columns:
                if normalization_method == 'zscore':
                    df_clean[f'{col}_normalized'] = zscore_normalize(df_clean, col)
                elif normalization_method == 'minmax':
                    df_clean[f'{col}_normalized'] = minmax_normalize(df_clean, col)
        
        report['normalized_columns'] = normalize_columns
        report['normalization_method'] = normalization_method
    
    report['final_shape'] = df_clean.shape
    report['remaining_columns'] = list(df_clean.columns)
    
    return df_clean, report

def validate_dataframe(dataframe, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: DataFrame to validate
        required_columns: List of required column names
        numeric_columns: List of columns that should be numeric
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in dataframe.columns:
                if not pd.api.types.is_numeric_dtype(dataframe[col]):
                    validation_results['is_valid'] = False
                    validation_results['errors'].append(f"Column '{col}' is not numeric")
    
    # Check for infinite values
    numeric_df = dataframe.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        infinite_count = np.isinf(numeric_df).sum().sum()
        if infinite_count > 0:
            validation_results['warnings'].append(f"Found {infinite_count} infinite values")
    
    # Check for duplicate rows
    duplicate_count = dataframe.duplicated().sum()
    if duplicate_count > 0:
        validation_results['warnings'].append(f"Found {duplicate_count} duplicate rows")
    
    return validation_resultsimport pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Load a CSV file, perform basic cleaning operations,
    and optionally save the cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove leading/trailing whitespace from string columns
        for col in categorical_cols:
            df[col] = df[col].str.strip()
        
        # Convert date columns if detected
        date_patterns = ['date', 'time', 'timestamp']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in date_patterns):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
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
    Validate dataframe structure and content.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    # Check for remaining null values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("Warning: DataFrame contains null values")
        for col, count in null_counts.items():
            if count > 0:
                print(f"  {col}: {count} nulls")
    
    return True

if __name__ == "__main__":
    # Example usage
    cleaned_df = clean_csv_data("input_data.csv", "cleaned_data.csv")
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        print(f"Data validation result: {is_valid}")