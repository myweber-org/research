
def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def z_score_normalize(data, column):
    """
    Normalize data using z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5, normalization_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    removal_stats = {}
    
    for col in numeric_columns:
        if col in df.columns:
            cleaned_df, removed = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            removal_stats[col] = removed
            
            if normalization_method == 'minmax':
                cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
            elif normalization_method == 'zscore':
                cleaned_df[f'{col}_normalized'] = z_score_normalize(cleaned_df, col)
    
    return cleaned_df, removal_stats

def validate_data(df, required_columns, min_rows=10):
    """
    Validate dataset structure and content
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(df) < min_rows:
        raise ValueError(f"Dataset has fewer than {min_rows} rows")
    
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0].index.tolist()
    
    validation_result = {
        'is_valid': len(missing_columns) == 0 and len(df) >= min_rows,
        'missing_columns': missing_columns,
        'row_count': len(df),
        'columns_with_nulls': columns_with_nulls,
        'null_counts': null_counts[null_counts > 0].to_dict()
    }
    
    return validation_result
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
    df.drop_duplicates(subset=[column_name], inplace=True, keep='first')
    df.reset_index(drop=True, inplace=True)
    return df

def normalize_string(text):
    """
    Normalize a string by removing extra spaces and special characters.
    """
    if not isinstance(text, str):
        return text
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def process_file(input_file, output_file, column_to_clean):
    """
    Read a CSV file, clean the specified column, and save to a new file.
    """
    try:
        df = pd.read_csv(input_file)
        df = clean_dataframe(df, column_to_clean)
        df[column_to_clean] = df[column_to_clean].apply(normalize_string)
        df.to_csv(output_file, index=False)
        print(f"Data cleaned and saved to {output_file}")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    target_column = "product_name"
    process_file(input_csv, output_csv, target_column)