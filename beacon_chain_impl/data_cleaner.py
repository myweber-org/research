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