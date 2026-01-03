import re
import unicodedata

def clean_text(text, remove_digits=False, keep_case=False):
    """
    Clean and normalize a given text string.

    Args:
        text (str): Input text to clean.
        remove_digits (bool): If True, remove all digits.
        keep_case (bool): If True, preserve original case.

    Returns:
        str: Cleaned text string.
    """
    if not isinstance(text, str):
        return ''

    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Optionally remove digits
    if remove_digits:
        text = re.sub(r'\d+', '', text)

    # Optionally convert to lowercase
    if not keep_case:
        text = text.lower()

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)

    return text.strip()

def tokenize_text(text, token_pattern=r'\b\w+\b'):
    """
    Tokenize text using a regular expression pattern.

    Args:
        text (str): Input text to tokenize.
        token_pattern (str): Regex pattern for tokenization.

    Returns:
        list: List of tokens.
    """
    cleaned = clean_text(text)
    tokens = re.findall(token_pattern, cleaned)
    return tokens

if __name__ == '__main__':
    sample = "Hello World! 123 This is a TEST.   "
    print(f"Original: '{sample}'")
    print(f"Cleaned: '{clean_text(sample)}'")
    print(f"Cleaned (no digits): '{clean_text(sample, remove_digits=True)}'")
    print(f"Tokens: {tokenize_text(sample)}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def remove_outliers_zscore(dataframe, column, threshold=3):
    z_scores = np.abs(stats.zscore(dataframe[column]))
    return dataframe[z_scores < threshold]

def normalize_minmax(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    if max_val == min_val:
        return dataframe[column].apply(lambda x: 0.0)
    return dataframe[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def normalize_zscore(dataframe, column):
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    if std_val == 0:
        return dataframe[column].apply(lambda x: 0.0)
    return dataframe[column].apply(lambda x: (x - mean_val) / std_val)

def clean_dataset(dataframe, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    df_clean = dataframe.copy()
    
    for col in numeric_columns:
        if col not in df_clean.columns:
            continue
            
        if outlier_method == 'iqr':
            df_clean = remove_outliers_iqr(df_clean, col)
        elif outlier_method == 'zscore':
            df_clean = remove_outliers_zscore(df_clean, col)
        
        if normalize_method == 'minmax':
            df_clean[col] = normalize_minmax(df_clean, col)
        elif normalize_method == 'zscore':
            df_clean[col] = normalize_zscore(df_clean, col)
    
    return df_clean.reset_index(drop=True)

def get_column_statistics(dataframe, column):
    if column not in dataframe.columns:
        return {}
    
    series = dataframe[column]
    return {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'count': series.count(),
        'missing': series.isnull().sum()
    }