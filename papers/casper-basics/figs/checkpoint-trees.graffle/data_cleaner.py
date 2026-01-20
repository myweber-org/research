
import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values with specified strategy
    """
    if columns is None:
        columns = data.columns
    
    data_clean = data.copy()
    
    for col in columns:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_clean[col].fillna(fill_value, inplace=True)
    
    return data_clean

def remove_duplicates(data, subset=None, keep='first'):
    """
    Remove duplicate rows from dataset
    """
    return data.drop_duplicates(subset=subset, keep=keep)

def normalize_data(data, columns=None, method='minmax'):
    """
    Normalize data using specified method
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    data_normalized = data.copy()
    
    for col in columns:
        if method == 'minmax':
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val != min_val:
                data_normalized[col] = (data[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val != 0:
                data_normalized[col] = (data[col] - mean_val) / std_val
        
        elif method == 'robust':
            median_val = data[col].median()
            iqr_val = stats.iqr(data[col])
            if iqr_val != 0:
                data_normalized[col] = (data[col] - median_val) / iqr_val
    
    return data_normalized

def clean_dataset(data, missing_strategy='mean', normalize_method=None, remove_outliers=False):
    """
    Comprehensive data cleaning pipeline
    """
    # Handle missing values
    cleaned_data = handle_missing_values(data, strategy=missing_strategy)
    
    # Remove duplicates
    cleaned_data = remove_duplicates(cleaned_data)
    
    # Remove outliers if requested
    if remove_outliers:
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outliers = detect_outliers_iqr(cleaned_data, col)
            cleaned_data = cleaned_data.drop(outliers.index)
    
    # Normalize if requested
    if normalize_method:
        cleaned_data = normalize_data(cleaned_data, method=normalize_method)
    
    return cleaned_data
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    df = pd.read_csv(input_file)
    
    df.replace('', np.nan, inplace=True)
    df.dropna(subset=['critical_column'], inplace=True)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_csv_data('raw_data.csv', 'cleaned_data.csv')
    print(f"Data cleaning complete. Shape: {cleaned_df.shape}")