import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling to range [0, 1].
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
    Normalize data using Z-score standardization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if col not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, col)
        
        if normalize_method == 'minmax':
            cleaned_data[col] = normalize_minmax(cleaned_data, col)
        elif normalize_method == 'zscore':
            cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

def validate_data(data, required_columns, numeric_threshold=0.8):
    """
    Validate dataset structure and content.
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) / len(data.columns) < numeric_threshold:
        print(f"Warning: Less than {numeric_threshold*100}% of columns are numeric")
    
    return True
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str): Strategy to fill missing values. 
                            Options: 'mean', 'median', 'mode', or 'drop'. 
                            Default is 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            elif fill_missing == 'mode':
                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
        print(f"Filled missing numeric values using {fill_missing}.")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        cleaned_df[col].fillna('Unknown', inplace=True)
    print("Filled missing categorical values with 'Unknown'.")
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate the dataset for required columns and basic integrity.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.empty:
        print("Dataset is empty.")
        return False
    
    print("Dataset validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, None, 15.3, 20.1, None],
        'category': ['A', 'B', None, 'A', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n")
    
    cleaned = clean_dataset(df, fill_missing='median')
    print("\nCleaned dataset:")
    print(cleaned)
    
    is_valid = validate_dataset(cleaned, required_columns=['id', 'value'])
    print(f"\nDataset valid: {is_valid}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process, defaults to all numeric columns
    factor (float): Multiplier for IQR, defaults to 1.5
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def remove_outliers_zscore(df, columns=None, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process
    threshold (float): Z-score threshold, defaults to 3
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            mask = z_scores < threshold
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize
    feature_range (tuple): Desired range of transformed data
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max - col_min != 0:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
                df_normalized[col] = df_normalized[col] * (max_val - min_val) + min_val
    
    return df_normalized

def normalize_zscore(df, columns=None):
    """
    Normalize data using Z-score standardization.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize
    
    Returns:
    pd.DataFrame: Dataframe with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val != 0:
                df_standardized[col] = (df[col] - mean_val) / std_val
    
    return df_standardized

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Dataframe with handled missing values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_processed = df.copy()
    
    if strategy == 'drop':
        df_processed = df_processed.dropna(subset=columns)
    else:
        for col in columns:
            if col in df.columns:
                if strategy == 'mean':
                    fill_value = df[col].mean()
                elif strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'mode':
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
                else:
                    fill_value = 0
                
                df_processed[col] = df_processed[col].fillna(fill_value)
    
    return df_processed.reset_index(drop=True)

def clean_data_pipeline(df, outlier_method='iqr', normalize_method='minmax', 
                       missing_strategy='mean', outlier_threshold=3):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    outlier_method (str): Outlier removal method ('iqr' or 'zscore')
    normalize_method (str): Normalization method ('minmax' or 'zscore')
    missing_strategy (str): Missing value handling strategy
    outlier_threshold (float): Threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned and normalized dataframe
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return df
    
    df_clean = handle_missing_values(df, strategy=missing_strategy, columns=numeric_cols)
    
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df_clean, columns=numeric_cols)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df_clean, columns=numeric_cols, threshold=outlier_threshold)
    
    if normalize_method == 'minmax':
        df_clean = normalize_minmax(df_clean, columns=numeric_cols)
    elif normalize_method == 'zscore':
        df_clean = normalize_zscore(df_clean, columns=numeric_cols)
    
    return df_clean

def validate_data(df, check_duplicates=True, check_infinite=True):
    """
    Validate data quality after cleaning.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    check_duplicates (bool): Whether to check for duplicate rows
    check_infinite (bool): Whether to check for infinite values
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': 0,
        'infinite_values': 0,
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    if check_duplicates:
        validation_results['duplicate_rows'] = df.duplicated().sum()
    
    if check_infinite:
        numeric_df = df.select_dtypes(include=[np.number])
        validation_results['infinite_values'] = np.isinf(numeric_df.values).sum()
    
    return validation_resultsimport pandas as pd
import numpy as np

def remove_missing_values(df, threshold=0.5):
    """
    Remove columns with missing values above threshold.
    """
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > threshold].index
    return df.drop(columns=columns_to_drop)

def normalize_numeric_columns(df, columns=None):
    """
    Normalize specified numeric columns to range [0,1].
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df[col] = (df[col] - col_min) / (col_max - col_min)
    return df

def encode_categorical(df, columns=None, method='onehot'):
    """
    Encode categorical columns using specified method.
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    if method == 'onehot':
        return pd.get_dummies(df, columns=columns, drop_first=True)
    elif method == 'label':
        encoded_df = df.copy()
        for col in columns:
            encoded_df[col] = pd.factorize(encoded_df[col])[0]
        return encoded_df
    else:
        raise ValueError("Method must be 'onehot' or 'label'")

def clean_dataset(filepath, output_path=None):
    """
    Main cleaning pipeline for CSV files.
    """
    df = pd.read_csv(filepath)
    
    df = remove_missing_values(df)
    df = normalize_numeric_columns(df)
    df = encode_categorical(df, method='onehot')
    
    if output_path:
        df.to_csv(output_path, index=False)
    
    return df