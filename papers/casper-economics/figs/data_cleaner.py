import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for identifying duplicates
        keep: determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_column(df, column_name):
    """
    Clean a numeric column by removing non-numeric values and converting to float.
    
    Args:
        df: pandas DataFrame
        column_name: name of the column to clean
    
    Returns:
        DataFrame with cleaned column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    original_dtype = df[column_name].dtype
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    
    converted_count = df[column_name].isna().sum() - df[column_name].isna().sum()
    if converted_count > 0:
        print(f"Converted {converted_count} values in column '{column_name}' from {original_dtype} to float")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
    
    Returns:
        Boolean indicating if validation passed
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 3, 1, 2, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'David'],
        'score': ['95', '88', '92', '95', '88', 'invalid']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_duplicates(df, subset=['id', 'name'])
    print("After removing duplicates:")
    print(cleaned_df)
    print()
    
    cleaned_df = clean_numeric_column(cleaned_df, 'score')
    print("After cleaning numeric column:")
    print(cleaned_df)
    print()
    
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'score'])
    print(f"DataFrame validation: {is_valid}")

if __name__ == "__main__":
    main()import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. 
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled. Default is None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and basic integrity.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_clean (list, optional): List of column names to apply string normalization.
                                       If None, all object dtype columns are cleaned.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = cleaned_df.shape[0]
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    removed_duplicates = initial_rows - cleaned_df.shape[0]
    
    # Determine which columns to normalize
    if columns_to_clean is None:
        # Select all object dtype columns (typically strings)
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    # Normalize string columns
    for col in columns_to_clean:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
    
    print(f"Removed {removed_duplicates} duplicate rows.")
    print(f"Normalized {len(columns_to_clean)} columns.")
    
    return cleaned_df

def _normalize_string(value):
    """
    Normalize a string value by converting to lowercase, removing extra whitespace,
    and stripping special characters from the edges.
    
    Parameters:
    value: Input value to normalize.
    
    Returns:
    Normalized string or original value if not a string.
    """
    if not isinstance(value, str):
        return value
    
    # Convert to lowercase
    normalized = value.lower()
    
    # Remove leading/trailing whitespace
    normalized = normalized.strip()
    
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    email_column (str): Name of the column containing email addresses.
    
    Returns:
    pd.DataFrame: DataFrame with additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    validated_df = df.copy()
    
    # Simple email validation regex
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validated_df['email_valid'] = validated_df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    
    valid_count = validated_df['email_valid'].sum()
    total_count = validated_df.shape[0]
    
    print(f"Email validation: {valid_count} valid out of {total_count} total emails.")
    
    return validated_df

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', '  ALICE WONDERLAND  '],
        'email': ['john@example.com', 'jane@example.com', 'invalid-email', 'alice@test.org'],
        'age': [25, 30, 25, 28]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    # Validate emails
    validated = validate_email_column(cleaned, 'email')
    print("\nDataFrame with email validation:")
    print(validated)
import pandas as pd
import numpy as np

def clean_data(file_path):
    """
    Load and clean a CSV file by removing duplicates,
    handling missing values, and standardizing formats.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Fill missing numeric values with column mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove leading/trailing whitespace from string columns
        for col in categorical_cols:
            df[col] = df[col].str.strip()
        
        # Convert date columns to datetime if possible
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='ignore')
            except:
                continue
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a new CSV file.
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return True
    return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_data(input_file)
    if cleaned_df is not None:
        save_cleaned_data(cleaned_df, output_file)