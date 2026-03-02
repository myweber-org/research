
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: Optional dictionary to rename columns
        drop_duplicates: Whether to remove duplicate rows
        normalize_text: Whether to normalize text columns (strip, lower case)
    
    Returns:
        Cleaned pandas DataFrame
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
        for column in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[column] = cleaned_df[column].apply(
                lambda x: str(x).strip().lower() if pd.notnull(x) else x
            )
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_email(email):
    """
    Validate email format using regex pattern.
    
    Args:
        email: Email string to validate
    
    Returns:
        Boolean indicating if email is valid
    """
    if pd.isna(email):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def filter_valid_emails(df, email_column='email'):
    """
    Filter DataFrame to only include rows with valid email addresses.
    
    Args:
        df: pandas DataFrame
        email_column: Name of the column containing email addresses
    
    Returns:
        DataFrame with only valid email rows
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    mask = df[email_column].apply(validate_email)
    valid_df = df[mask].copy()
    invalid_count = len(df) - len(valid_df)
    
    print(f"Filtered out {invalid_count} rows with invalid email addresses")
    return valid_df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a numeric column using the Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        column: Name of the numeric column
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    filtered_df = df[mask].copy()
    outliers_removed = len(df) - len(filtered_df)
    
    print(f"Removed {outliers_removed} outliers from column '{column}'")
    return filtered_df

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Args:
        df: pandas DataFrame to save
        output_path: Path to save the file
        format: File format ('csv', 'excel', or 'parquet')
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv', 'excel', or 'parquet'")
    
    print(f"Saved cleaned data to {output_path}")