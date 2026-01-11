
import pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns=None, method='zscore'):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    if method == 'zscore':
        for col in columns:
            if col in df.columns:
                df_normalized[col] = stats.zscore(df[col])
    elif method == 'minmax':
        for col in columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    elif method == 'robust':
        for col in columns:
            if col in df.columns:
                median = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                if iqr > 0:
                    df_normalized[col] = (df[col] - median) / iqr
    
    return df_normalized

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    if method == 'iqr':
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                df_clean = df_clean[mask]
    
    elif method == 'zscore':
        for col in columns:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                mask = z_scores < threshold
                df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def handle_missing_values(df, columns=None, strategy='mean'):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'ffill':
                df_filled[col] = df[col].fillna(method='ffill')
                continue
            elif strategy == 'bfill':
                df_filled[col] = df[col].fillna(method='bfill')
                continue
            else:
                fill_value = 0
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def clean_dataset(df, numeric_columns=None, normalize=True, remove_outliers_flag=True, handle_missing=True):
    df_processed = df.copy()
    
    if handle_missing:
        df_processed = handle_missing_values(df_processed, numeric_columns)
    
    if remove_outliers_flag and numeric_columns:
        df_processed = remove_outliers(df_processed, numeric_columns)
    
    if normalize and numeric_columns:
        df_processed = normalize_data(df_processed, numeric_columns)
    
    return df_processed

def validate_data(df, check_duplicates=True, check_types=True):
    validation_report = {}
    
    validation_report['rows'] = len(df)
    validation_report['columns'] = len(df.columns)
    validation_report['missing_values'] = df.isnull().sum().sum()
    
    if check_duplicates:
        validation_report['duplicate_rows'] = df.duplicated().sum()
    
    if check_types:
        type_counts = df.dtypes.value_counts().to_dict()
        validation_report['data_types'] = {str(k): int(v) for k, v in type_counts.items()}
    
    return validation_report
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'drop',
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    input_path: Path to input CSV file
    output_path: Path to save cleaned CSV file
    missing_strategy: Strategy for handling missing values ('drop', 'fill', 'interpolate')
    fill_value: Value to fill missing data with (if strategy is 'fill')
    
    Returns:
    Cleaned DataFrame
    """
    
    df = pd.read_csv(input_path)
    
    original_rows = len(df)
    print(f"Original data: {original_rows} rows, {len(df.columns)} columns")
    
    df = df.drop_duplicates()
    
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy == 'fill':
        if fill_value is not None:
            df = df.fillna(fill_value)
        else:
            df = df.fillna(df.mean(numeric_only=True))
    elif missing_strategy == 'interpolate':
        df = df.interpolate(method='linear', limit_direction='forward')
    
    cleaned_rows = len(df)
    print(f"Cleaned data: {cleaned_rows} rows, {len(df.columns)} columns")
    print(f"Removed {original_rows - cleaned_rows} rows")
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    
    return df

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Detect outliers using IQR method for a specific column.
    
    Parameters:
    df: DataFrame containing the data
    column: Column name to check for outliers
    
    Returns:
    Boolean Series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a column using min-max scaling.
    
    Parameters:
    df: DataFrame containing the data
    column: Column name to normalize
    
    Returns:
    DataFrame with normalized column
    """
    if column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()
        
        if max_val != min_val:
            df[f'{column}_normalized'] = (df[column] - min_val) / (max_val - min_val)
        else:
            df[f'{column}_normalized'] = 0
    
    return df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 5],
        'B': [10, 20, 30, np.nan, 50, 10],
        'C': [100, 200, 300, 400, 500, 100]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data(
        'test_data.csv',
        'cleaned_data.csv',
        missing_strategy='fill'
    )
    
    outliers = detect_outliers_iqr(cleaned_df, 'C')
    print(f"Outliers in column C: {outliers.sum()}")
    
    normalized_df = normalize_column(cleaned_df, 'B')
    print(normalized_df.head())