import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing data in a DataFrame using specified strategy.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values.
                        Options: 'mean', 'median', 'mode', 'drop', 'fill_zero'
        columns (list): List of columns to apply cleaning to. If None, applies to all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_cleaned = df.copy()
    
    if columns is None:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    if strategy == 'drop':
        df_cleaned = df_cleaned.dropna(subset=columns)
    elif strategy == 'mean':
        for col in columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
    elif strategy == 'median':
        for col in columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    elif strategy == 'mode':
        for col in columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 0)
    elif strategy == 'fill_zero':
        for col in columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].fillna(0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_cleaned

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outlier information
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    outlier_report = pd.DataFrame({
        'column': [column],
        'total_values': [len(df)],
        'outlier_count': [len(outliers)],
        'outlier_percentage': [len(outliers) / len(df) * 100],
        'lower_bound': [lower_bound],
        'upper_bound': [upper_bound]
    })
    
    return outlier_report, outliers

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to normalize
        method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df_normalized[column].min()
        max_val = df_normalized[column].max()
        if max_val != min_val:
            df_normalized[f'{column}_normalized'] = (df_normalized[column] - min_val) / (max_val - min_val)
        else:
            df_normalized[f'{column}_normalized'] = 0.5
    elif method == 'zscore':
        mean_val = df_normalized[column].mean()
        std_val = df_normalized[column].std()
        if std_val != 0:
            df_normalized[f'{column}_normalized'] = (df_normalized[column] - mean_val) / std_val
        else:
            df_normalized[f'{column}_normalized'] = 0
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df_normalized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (mean imputation):")
    cleaned_df = clean_missing_data(df, strategy='mean')
    print(cleaned_df)
    
    print("\nOutlier detection for column 'C':")
    report, outliers = detect_outliers_iqr(cleaned_df, 'C')
    print(report)
    
    print("\nNormalized column 'C':")
    normalized_df = normalize_column(cleaned_df, 'C', method='minmax')
    print(normalized_df[['C', 'C_normalized']])
    
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
    print(f"\nDataFrame validation: {is_valid} - {message}")