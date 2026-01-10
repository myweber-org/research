import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean dataset by handling missing values and outliers.
    
    Args:
        df: pandas DataFrame
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        outlier_method: Method for handling outliers ('iqr', 'zscore')
        columns: List of columns to clean, if None clean all numeric columns
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if missing_strategy != 'drop':
            if missing_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif missing_strategy == 'median':
                fill_value = df_clean[col].median()
            elif missing_strategy == 'mode':
                fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else np.nan
            else:
                fill_value = 0
                
            df_clean[col].fillna(fill_value, inplace=True)
        else:
            df_clean = df_clean.dropna(subset=[col])
        
        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        elif outlier_method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df_clean[col]))
            df_clean = df_clean[z_scores < 3]
    
    return df_clean

def normalize_data(df, method='minmax', columns=None):
    """
    Normalize data using specified method.
    
    Args:
        df: pandas DataFrame
        method: Normalization method ('minmax', 'standard')
        columns: List of columns to normalize
    
    Returns:
        Normalized pandas DataFrame
    """
    df_norm = df.copy()
    
    if columns is None:
        columns = df_norm.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df_norm.columns:
            continue
            
        if method == 'minmax':
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val != min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        
        elif method == 'standard':
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val != 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm

def validate_data(df, check_missing=True, check_duplicates=True):
    """
    Validate data quality.
    
    Args:
        df: pandas DataFrame
        check_missing: Check for missing values
        check_duplicates: Check for duplicate rows
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    if check_missing:
        missing_counts = df.isnull().sum()
        missing_percentage = (missing_counts / len(df)) * 100
        validation_results['missing_values'] = {
            'counts': missing_counts.to_dict(),
            'percentage': missing_percentage.to_dict()
        }
    
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        validation_results['duplicates'] = {
            'count': duplicate_count,
            'percentage': (duplicate_count / len(df)) * 100 if len(df) > 0 else 0
        }
    
    validation_results['summary'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_results

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, 8, 9],
        'C': [10, 11, 12, 13, 14]
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned_data = clean_dataset(sample_data, missing_strategy='mean', outlier_method='iqr')
    print("\nCleaned data:")
    print(cleaned_data)
    
    normalized_data = normalize_data(cleaned_data, method='minmax')
    print("\nNormalized data:")
    print(normalized_data)
    
    validation = validate_data(sample_data)
    print("\nValidation results:")
    print(validation)