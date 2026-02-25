import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fillna_strategy='mean', columns=None):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fillna_strategy (str): Strategy for filling NaN values.
                               Options: 'mean', 'median', 'mode', 'zero', 'ffill'.
        columns (list): Specific columns to apply cleaning. If None, applies to all.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns.tolist()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    for col in columns:
        if col in df_clean.columns:
            if df_clean[col].dtype in [np.float64, np.int64]:
                if fillna_strategy == 'mean':
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif fillna_strategy == 'median':
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif fillna_strategy == 'zero':
                    df_clean[col].fillna(0, inplace=True)
                elif fillna_strategy == 'ffill':
                    df_clean[col].fillna(method='ffill', inplace=True)
            elif df_clean[col].dtype == object:
                if fillna_strategy == 'mode':
                    mode_val = df_clean[col].mode()
                    df_clean[col].fillna(mode_val[0] if not mode_val.empty else '', inplace=True)
                elif fillna_strategy == 'ffill':
                    df_clean[col].fillna(method='ffill', inplace=True)
                else:
                    df_clean[col].fillna('', inplace=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): Columns to check for outliers. If None, uses all numeric columns.
        multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    mask = pd.Series([True] * len(df_clean))
    
    for col in columns:
        if col in df_clean.columns and df_clean[col].dtype in [np.float64, np.int64]:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            col_mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            mask = mask & col_mask
    
    return df_clean[mask].reset_index(drop=True)import pandas as pd

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
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value:.2f}")
    
    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if df.empty:
        print("Validation failed: DataFrame is empty.")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 4],
        'B': [5, None, 5, 6, 7, 8],
        'C': ['x', 'y', 'x', 'z', 'w', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nData is valid: {is_valid}")
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def remove_outliers_zscore(df, columns, threshold=3):
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            z_scores = np.abs(stats.zscore(cleaned_df[col]))
            cleaned_df = cleaned_df[z_scores < threshold]
    return cleaned_df

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def normalize_zscore(df, columns):
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            if std_val > 0:
                normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    return normalized_df

def handle_missing_values(df, strategy='mean'):
    handled_df = df.copy()
    numeric_cols = handled_df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            handled_df[col].fillna(handled_df[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            handled_df[col].fillna(handled_df[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            handled_df[col].fillna(handled_df[col].mode()[0], inplace=True)
    elif strategy == 'drop':
        handled_df.dropna(subset=numeric_cols, inplace=True)
    
    return handled_df

def clean_dataset(input_file, output_file, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    df = load_dataset(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if outlier_method == 'iqr':
        df = remove_outliers_iqr(df, numeric_columns)
    elif outlier_method == 'zscore':
        df = remove_outliers_zscore(df, numeric_columns)
    
    df = handle_missing_values(df, strategy=missing_strategy)
    
    if normalize_method == 'minmax':
        df = normalize_minmax(df, numeric_columns)
    elif normalize_method == 'zscore':
        df = normalize_zscore(df, numeric_columns)
    
    df.to_csv(output_file, index=False)
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv', 'cleaned_data.csv')
    print(f"Dataset cleaned. Shape: {cleaned_data.shape}")