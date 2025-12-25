
import pandas as pd
import re

def clean_dataset(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping original column names to new names
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip whitespace, lowercase)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of columns that must be present
        numeric_columns (list): List of columns that should be numeric
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'non_numeric_columns': [],
        'null_counts': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    validation_results['non_numeric_columns'].append(col)
                    validation_results['is_valid'] = False
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_results['null_counts'][col] = null_count
    
    return validation_results

def sample_data(df, sample_size=1000, random_state=42):
    """
    Sample data from DataFrame for testing purposes.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        sample_size (int): Number of rows to sample
        random_state (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Sampled DataFrame
    """
    if len(df) <= sample_size:
        return df
    
    return df.sample(n=min(sample_size, len(df)), random_state=random_state)
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(input_file, output_file, outlier_method='iqr', normalize_method='minmax'):
    df = load_data(input_file)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            df = remove_outliers_iqr(df, col)
        elif outlier_method == 'zscore':
            df = remove_outliers_zscore(df, col)
    
    for col in numeric_cols:
        if normalize_method == 'minmax':
            df = normalize_minmax(df, col)
        elif normalize_method == 'zscore':
            df = normalize_zscore(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val - min_val == 0:
        return data[column].apply(lambda x: 0.5)
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization_method='minmax'):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
        if outlier_removal:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        if normalization_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalization_method == 'zscore':
            cleaned_df[col] = standardize_zscore(cleaned_df, col)
    return cleaned_df

def validate_dataframe(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("DataFrame is empty")
    return True
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(df, numeric_columns, method='iqr', normalize=True):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    if normalize:
        for col in numeric_columns:
            cleaned_df = normalize_minmax(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def get_summary_statistics(df):
    summary = pd.DataFrame({
        'mean': df.mean(),
        'median': df.median(),
        'std': df.std(),
        'min': df.min(),
        'max': df.max(),
        'count': df.count()
    })
    return summary.T