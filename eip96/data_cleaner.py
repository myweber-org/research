
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for identifying duplicates
        keep: determines which duplicates to keep ('first', 'last', or False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
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

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    df = df.dropna()
    return df

def main():
    cleaned_data = clean_dataset('raw_data.csv', ['age', 'income', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Cleaned data shape: {cleaned_data.shape}")

if __name__ == "__main__":
    main()
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_missing == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_missing == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fill_missing == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    else:
        raise ValueError("fill_missing must be 'mean', 'median', 'mode', or 'drop'")
    
    # Remove duplicates
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, check_missing=True, check_duplicates=True):
    """
    Validate a DataFrame by checking for missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    check_missing (bool): Whether to check for missing values.
    check_duplicates (bool): Whether to check for duplicate rows.
    
    Returns:
    dict: Dictionary containing validation results.
    """
    validation_results = {}
    
    if check_missing:
        missing_count = df.isnull().sum().sum()
        validation_results['missing_values'] = missing_count
    
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        validation_results['duplicate_rows'] = duplicate_count
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and duplicates
    data = {
        'A': [1, 2, None, 4, 2],
        'B': [5, None, 7, 8, 5],
        'C': [9, 10, 11, None, 9]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation results:")
    print(validate_dataset(df))
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nValidation results after cleaning:")
    print(validate_dataset(cleaned))import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using IQR method.
    Returns boolean mask for outliers.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, columns, threshold=1.5):
    """
    Remove outliers from specified columns.
    """
    clean_data = data.copy()
    for col in columns:
        if col in clean_data.columns:
            outliers = detect_outliers_iqr(clean_data, col, threshold)
            clean_data = clean_data[~outliers]
    return clean_data.reset_index(drop=True)

def normalize_minmax(data, columns):
    """
    Normalize data using min-max scaling.
    """
    normalized_data = data.copy()
    for col in columns:
        if col in normalized_data.columns:
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            if max_val > min_val:
                normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
    return normalized_data

def standardize_zscore(data, columns):
    """
    Standardize data using z-score normalization.
    """
    standardized_data = data.copy()
    for col in columns:
        if col in standardized_data.columns:
            mean_val = standardized_data[col].mean()
            std_val = standardized_data[col].std()
            if std_val > 0:
                standardized_data[col] = (standardized_data[col] - mean_val) / std_val
    return standardized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    """
    filled_data = data.copy()
    if columns is None:
        columns = filled_data.columns
    
    for col in columns:
        if col in filled_data.columns and filled_data[col].isnull().any():
            if strategy == 'mean':
                fill_value = filled_data[col].mean()
            elif strategy == 'median':
                fill_value = filled_data[col].median()
            elif strategy == 'mode':
                fill_value = filled_data[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                continue
            
            filled_data[col] = filled_data[col].fillna(fill_value)
    
    return filled_data

def clean_dataset(data, numeric_columns, outlier_threshold=1.5, 
                  normalization='minmax', missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    """
    # Handle missing values
    cleaned_data = handle_missing_values(data, strategy=missing_strategy, 
                                         columns=numeric_columns)
    
    # Remove outliers
    cleaned_data = remove_outliers(cleaned_data, numeric_columns, 
                                   threshold=outlier_threshold)
    
    # Apply normalization
    if normalization == 'minmax':
        cleaned_data = normalize_minmax(cleaned_data, numeric_columns)
    elif normalization == 'zscore':
        cleaned_data = standardize_zscore(cleaned_data, numeric_columns)
    
    return cleaned_data

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'feature3': np.random.uniform(0, 1, 100)
    })
    
    # Add some outliers and missing values
    sample_data.loc[10, 'feature1'] = 500
    sample_data.loc[20, 'feature2'] = 1000
    sample_data.loc[30, 'feature3'] = np.nan
    
    # Clean the data
    numeric_cols = ['feature1', 'feature2', 'feature3']
    cleaned = clean_dataset(sample_data, numeric_cols, 
                           outlier_threshold=1.5,
                           normalization='zscore',
                           missing_strategy='mean')
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Missing values in cleaned data: {cleaned.isnull().sum().sum()}")