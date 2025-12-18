
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list, optional): List of column names to check for duplicates.
                                       If None, checks all columns.
    fill_missing (str or dict): Method to fill missing values.
                                Can be 'mean', 'median', 'mode', or a dictionary
                                of column:value pairs for custom filling.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns.tolist()
    
    cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check, keep='first')
    
    # Handle missing values
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().any():
            if isinstance(fill_missing, dict):
                # Use custom value if provided in dictionary
                if column in fill_missing:
                    cleaned_df[column] = cleaned_df[column].fillna(fill_missing[column])
                else:
                    # If column not in dictionary, use column mean for numeric, mode for categorical
                    if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
                    else:
                        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
            elif fill_missing == 'mean':
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
                else:
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
            elif fill_missing == 'median':
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
                else:
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
            elif fill_missing == 'mode':
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
            else:
                # Default to mean for numeric, mode for categorical
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
                else:
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Dataset is valid"