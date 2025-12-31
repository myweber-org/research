
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
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

def clean_dataset(file_path, output_path):
    """
    Load a dataset, clean specified columns, and save the cleaned data.
    
    Parameters:
    file_path (str): Path to the input CSV file.
    output_path (str): Path to save the cleaned CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original dataset shape: {df.shape}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            df = remove_outliers_iqr(df, col)
        
        print(f"Cleaned dataset shape: {df.shape}")
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [3, 1, 2, 1, 4, 3, 5, 2]
    cleaned = remove_duplicates_preserve_order(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_mapping (dict, optional): Dictionary mapping old column names to new ones.
        drop_duplicates (bool): Whether to remove duplicate rows.
        normalize_text (bool): Whether to normalize text in object columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
    
    return cleaned_df

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Args:
        text (str): Input string.
    
    Returns:
        str: Normalized string.
    """
    if not isinstance(text, str):
        return text
    
    normalized = text.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    return normalized

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and non-null values.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        return False, f"Found null values in columns: {null_counts[null_counts > 0].to_dict()}"
    
    return True, "DataFrame is valid"import pandas as pd

def clean_dataframe(df):
    """
    Remove duplicate rows and normalize column names.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    
    # Normalize column names: strip whitespace, lowercase, replace spaces with underscores
    df_cleaned.columns = (
        df_cleaned.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
    )
    
    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and required columns.
    """
    if required_columns is None:
        required_columns = []
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Product Name': ['A', 'B', 'A', 'C'],
        'Price': [100, 200, 100, 300],
        'Category ': ['X', 'Y', 'X', 'Z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataframe(df)
    print(cleaned_df)
    
    try:
        validate_dataframe(cleaned_df, ['product_name', 'price'])
        print("\nData validation passed.")
    except ValueError as e:
        print(f"\nValidation error: {e}")
import csv
import os

def clean_csv(input_file, output_file, remove_duplicates=True, strip_whitespace=True):
    """
    Clean a CSV file by removing duplicates and stripping whitespace.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output cleaned CSV file.
        remove_duplicates (bool): Whether to remove duplicate rows.
        strip_whitespace (bool): Whether to strip whitespace from all fields.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
    
    seen_rows = set()
    cleaned_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        cleaned_rows.append(header)
        
        for row in reader:
            if strip_whitespace:
                row = [field.strip() if isinstance(field, str) else field for field in row]
            
            row_tuple = tuple(row)
            
            if remove_duplicates:
                if row_tuple in seen_rows:
                    continue
                seen_rows.add(row_tuple)
            
            cleaned_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)
    
    print(f"Cleaned data saved to '{output_file}'")
    print(f"Original rows: {len(cleaned_rows) + len(seen_rows) - 1}")
    print(f"Cleaned rows: {len(cleaned_rows) - 1}")

def validate_csv(file_path, required_columns=None):
    """
    Validate a CSV file for basic structure and required columns.
    
    Args:
        file_path (str): Path to the CSV file.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found.")
        return False
    
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)
            
            if required_columns:
                missing_columns = [col for col in required_columns if col not in header]
                if missing_columns:
                    print(f"Missing required columns: {missing_columns}")
                    return False
            
            row_count = 0
            for row in reader:
                row_count += 1
                if len(row) != len(header):
                    print(f"Row {row_count + 1} has {len(row)} columns, expected {len(header)}")
                    return False
            
            print(f"CSV validation passed: {row_count} rows, {len(header)} columns")
            return True
            
    except Exception as e:
        print(f"Error validating CSV: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    sample_input = "sample_data.csv"
    sample_output = "cleaned_data.csv"
    
    # Create a sample CSV file for testing
    sample_data = [
        ["Name", "Age", "Email"],
        ["John Doe", "30", "john@example.com"],
        ["Jane Smith", "25", "jane@example.com"],
        ["John Doe", "30", "john@example.com"],  # Duplicate
        [" Bob Johnson ", " 35 ", "bob@example.com"]  # With whitespace
    ]
    
    with open(sample_input, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(sample_data)
    
    print("Created sample CSV file")
    
    # Clean the CSV
    clean_csv(sample_input, sample_output)
    
    # Validate the cleaned CSV
    validate_csv(sample_output, required_columns=["Name", "Email"])
    
    # Clean up sample files
    os.remove(sample_input)
    os.remove(sample_output)
    print("Sample files cleaned up")