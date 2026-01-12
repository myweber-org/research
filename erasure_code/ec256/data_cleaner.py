
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict): Optional mapping to rename columns
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns
    
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
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('DataFrame is empty')
        return validation_results
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    null_counts = df.isnull().sum()
    if null_counts.any():
        null_info = null_counts[null_counts > 0].to_dict()
        validation_results['warnings'].append(f'Columns with null values: {null_info}')
    
    return validation_results

def sample_data_cleaning():
    """
    Example usage of the data cleaning functions.
    """
    sample_data = {
        'Name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', 'Alice Brown'],
        'Email': ['john@example.com', 'jane@example.com', 'john@example.com', 'bob@example.com', 'alice@example.com'],
        'Age': [25, 30, 25, 35, 28],
        'City': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Boston']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataframe(df, drop_duplicates=True, normalize_text=True)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataframe(cleaned_df, required_columns=['Name', 'Email', 'Age'])
    print("Validation Results:")
    print(validation)

if __name__ == "__main__":
    sample_data_cleaning()