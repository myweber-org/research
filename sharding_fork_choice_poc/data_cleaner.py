
import pandas as pd
import numpy as np

def clean_dataset(df, text_columns=None, fill_na=True):
    """
    Clean a pandas DataFrame by handling missing values and standardizing text.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    text_columns (list): List of column names containing text data
    fill_na (bool): Whether to fill numerical NA values with column median
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values for numerical columns
    if fill_na:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Standardize text columns
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
                df_clean[col] = df_clean[col].replace({'nan': '', 'none': '', 'null': ''})
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results with status and messages
    """
    validation_result = {
        'is_valid': True,
        'messages': [],
        'missing_columns': []
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_result['is_valid'] = False
        validation_result['messages'].append('Input is not a pandas DataFrame')
        return validation_result
    
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['messages'].append('DataFrame is empty')
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing
            validation_result['messages'].append(f'Missing required columns: {missing}')
    
    return validation_result

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'name': ['Alice', 'BOB', None, '  Charlie  '],
#         'age': [25, None, 30, 35],
#         'score': [85.5, 92.0, None, 78.5]
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned_df = clean_dataset(df, text_columns=['name'])
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)
#     
#     validation = validate_dataframe(cleaned_df, required_columns=['name', 'age'])
#     print("\nValidation Result:")
#     print(validation)