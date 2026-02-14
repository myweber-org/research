
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_normalization='lower'):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns_to_clean (list, optional): List of column names to normalize. If None, all object dtype columns are used.
    remove_duplicates (bool): If True, remove duplicate rows.
    case_normalization (str): One of 'lower', 'upper', or None to specify case normalization.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()

    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")

    if columns_to_clean is None:
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns.tolist()

    for col in columns_to_clean:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].astype(str)
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))

            if case_normalization == 'lower':
                cleaned_df[col] = cleaned_df[col].str.lower()
            elif case_normalization == 'upper':
                cleaned_df[col] = cleaned_df[col].str.upper()

            print(f"Cleaned column: {col}")

    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    email_column (str): Name of the column containing email addresses.

    Returns:
    pd.DataFrame: DataFrame with an additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")

    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].astype(str).str.match(email_pattern)
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['Alice', 'Bob', 'Alice', '  charlie  ', 'DAVE'],
        'email': ['alice@example.com', 'invalid-email', 'alice@example.com', 'charlie@test.org', 'DAVE@DOMAIN.COM'],
        'value': [1, 2, 1, 3, 4]
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)

    cleaned = clean_dataframe(df, case_normalization='lower')
    print("\nCleaned DataFrame:")
    print(cleaned)

    validated = validate_email_column(cleaned, 'email')
    print("\nDataFrame with email validation:")
    print(validated)
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean', output_path=None):
    """
    Load a CSV file, handle missing values, and optionally save cleaned data.
    
    Parameters:
    file_path (str): Path to the input CSV file.
    fill_method (str): Method to fill missing values ('mean', 'median', 'mode', 'zero').
    output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
    
    Returns:
    pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values.")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if fill_method == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif fill_method == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif fill_method == 'mode':
                for col in numeric_cols:
                    df[col] = df[col].fillna(df[col].mode()[0])
            elif fill_method == 'zero':
                df[numeric_cols] = df[numeric_cols].fillna(0)
            else:
                raise ValueError("Invalid fill_method. Choose from 'mean', 'median', 'mode', 'zero'.")
            
            print(f"Missing values filled using '{fill_method}' method.")
        else:
            print("No missing values found.")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_method='mean')
    
    if cleaned_df is not None:
        validation_result = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
        print(f"Validation result: {validation_result}")
        print("Cleaned DataFrame:")
        print(cleaned_df)