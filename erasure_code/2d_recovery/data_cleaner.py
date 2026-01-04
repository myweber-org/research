import pandas as pd

def clean_dataset(df, columns_to_check=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df: pandas DataFrame to clean.
        columns_to_check: List of columns to check for duplicates. If None, checks all columns.
        drop_duplicates: Boolean indicating whether to drop duplicate rows.
        fill_missing: Method to fill missing values ('mean', 'median', 'mode', or a scalar value).
    
    Returns:
        Cleaned pandas DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        if columns_to_check:
            cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
        else:
            cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
                elif isinstance(fill_missing, (int, float, str)):
                    cleaned_df[column].fillna(fill_missing, inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df: pandas DataFrame to validate.
        required_columns: List of column names that must be present.
        min_rows: Minimum number of rows required.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"