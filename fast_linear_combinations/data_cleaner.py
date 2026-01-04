
import pandas as pd
import re

def clean_dataframe(df, column_name):
    """
    Clean a specific column in a DataFrame by removing duplicates,
    stripping whitespace, and converting to lowercase.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].str.lower()
    df = df.drop_duplicates(subset=[column_name], keep='first')
    
    return df

def normalize_string(text):
    """
    Normalize a string by removing special characters and extra spaces.
    """
    if not isinstance(text, str):
        return text
    
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def process_csv(input_path, output_path, column_to_clean):
    """
    Read a CSV file, clean the specified column, and save the result.
    """
    df = pd.read_csv(input_path)
    df = clean_dataframe(df, column_to_clean)
    df[column_to_clean] = df[column_to_clean].apply(normalize_string)
    df.to_csv(output_path, index=False)
    
    return df

if __name__ == "__main__":
    input_file = "input_data.csv"
    output_file = "cleaned_data.csv"
    target_column = "product_name"
    
    try:
        result_df = process_csv(input_file, output_file, target_column)
        print(f"Data cleaned successfully. Rows processed: {len(result_df)}")
    except Exception as e:
        print(f"Error during processing: {e}")
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

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_data(data, column):
    """
    Standardize data using Z-score normalization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values with different strategies
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for column in columns:
        if data_copy[column].isnull().any():
            if strategy == 'mean':
                fill_value = data_copy[column].mean()
            elif strategy == 'median':
                fill_value = data_copy[column].median()
            elif strategy == 'mode':
                fill_value = data_copy[column].mode()[0]
            elif strategy == 'ffill':
                data_copy[column] = data_copy[column].fillna(method='ffill')
                continue
            elif strategy == 'bfill':
                data_copy[column] = data_copy[column].fillna(method='bfill')
                continue
            else:
                fill_value = 0
            
            data_copy[column] = data_copy[column].fillna(fill_value)
    
    return data_copy

def clean_dataset(data, outlier_method='zscore', normalize=False, standardize=False, missing_strategy='mean'):
    """
    Main function to clean dataset with multiple options
    """
    cleaned_data = data.copy()
    
    numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        if outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        elif outlier_method == 'iqr':
            outliers = detect_outliers_iqr(cleaned_data, column)
            cleaned_data = cleaned_data.drop(outliers.index)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    if normalize:
        for column in numeric_columns:
            if column in cleaned_data.columns:
                cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
    
    if standardize:
        for column in numeric_columns:
            if column in cleaned_data.columns:
                cleaned_data[f'{column}_standardized'] = standardize_data(cleaned_data, column)
    
    return cleaned_data.reset_index(drop=True)