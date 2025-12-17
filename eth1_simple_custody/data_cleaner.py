
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean dataset by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    missing_strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for outlier detection ('iqr', 'zscore')
    columns (list): Specific columns to clean, if None clean all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].dtype in [np.float64, np.int64]:
            handle_missing_values(df_clean, col, missing_strategy)
            handle_outliers(df_clean, col, outlier_method)
    
    return df_clean

def handle_missing_values(df, column, strategy):
    """Handle missing values in specified column."""
    if strategy == 'mean':
        fill_value = df[column].mean()
    elif strategy == 'median':
        fill_value = df[column].median()
    elif strategy == 'mode':
        fill_value = df[column].mode()[0] if not df[column].mode().empty else 0
    elif strategy == 'drop':
        df.dropna(subset=[column], inplace=True)
        return
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")
    
    df[column].fillna(fill_value, inplace=True)

def handle_outliers(df, column, method):
    """Handle outliers in specified column."""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        median_val = df[column].median()
        df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), 
                             median_val, df[column])
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        z_scores = np.abs((df[column] - mean_val) / std_val)
        
        median_val = df[column].median()
        df[column] = np.where(z_scores > 3, median_val, df[column])
    
    else:
        raise ValueError(f"Unknown outlier method: {method}")

def get_data_summary(df):
    """Generate summary statistics for the dataframe."""
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['a', 'b', 'c', 'd', 'e']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original Data:")
    print(df)
    print("\nSummary:")
    print(get_data_summary(df))
    
    cleaned_df = clean_dataset(df, missing_strategy='median', outlier_method='iqr')
    print("\nCleaned Data:")
    print(cleaned_df)