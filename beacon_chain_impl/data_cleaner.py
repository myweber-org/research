import re
from typing import List, Optional

def remove_special_characters(text: str, keep_spaces: bool = True) -> str:
    """
    Remove all non-alphanumeric characters from the input string.
    Optionally preserve spaces.
    """
    if keep_spaces:
        pattern = r'[^A-Za-z0-9\s]'
    else:
        pattern = r'[^A-Za-z0-9]'
    return re.sub(pattern, '', text)

def normalize_whitespace(text: str) -> str:
    """
    Replace multiple whitespace characters with a single space.
    Also strip leading and trailing whitespace.
    """
    return ' '.join(text.split())

def tokenize_text(text: str, lowercase: bool = True) -> List[str]:
    """
    Split text into tokens (words). Optionally convert to lowercase.
    """
    if lowercase:
        text = text.lower()
    tokens = text.split()
    return tokens

def clean_text_pipeline(
    text: str,
    remove_special: bool = True,
    normalize_ws: bool = True,
    tokenize: bool = False
) -> Optional[str]:
    """
    Apply a series of cleaning operations to the input text.
    Returns cleaned string or token list based on parameters.
    """
    if not isinstance(text, str):
        return None

    cleaned = text

    if remove_special:
        cleaned = remove_special_characters(cleaned)

    if normalize_ws:
        cleaned = normalize_whitespace(cleaned)

    if tokenize:
        return tokenize_text(cleaned)

    return cleaned

def batch_clean_texts(texts: List[str], **kwargs) -> List[Optional[str]]:
    """
    Apply cleaning pipeline to a list of text strings.
    Returns list of cleaned texts (or None for invalid inputs).
    """
    return [clean_text_pipeline(text, **kwargs) for text in texts]
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(file_path, output_path=None):
    """
    Load a dataset, clean specified columns, and optionally save the result.
    
    Parameters:
    file_path (str): Path to the input CSV file.
    output_path (str, optional): Path to save the cleaned CSV file.
    
    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_data = clean_dataset(input_file, output_file)
        print(f"Original data shape: {pd.read_csv(input_file).shape}")
        print(f"Cleaned data shape: {cleaned_data.shape}")
        print("Data cleaning completed successfully.")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns):
    """Normalize data using min-max scaling."""
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def handle_missing_values(df, strategy='mean'):
    """Handle missing values with specified strategy."""
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
    
    return df_filled

def clean_dataset(filepath, numeric_columns):
    """Main function to clean dataset."""
    df = load_dataset(filepath)
    df = handle_missing_values(df, strategy='median')
    df = remove_outliers_iqr(df, numeric_columns)
    df = normalize_data(df, numeric_columns)
    return df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature3': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_dataset('sample_data.csv', ['feature1', 'feature2', 'feature3'])
    print("Cleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned data summary:")
    print(cleaned_df.describe())