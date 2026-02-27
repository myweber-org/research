
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    if columns_to_clean is None:
        # Identify string columns automatically
        columns_to_clean = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_clean:
        if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].apply(normalize_string)
    
    return df_cleaned

def normalize_string(s):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(s):
        return s
    
    # Convert to string if not already
    s = str(s)
    
    # Convert to lowercase
    s = s.lower()
    
    # Remove extra whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    # Remove special characters (keep alphanumeric and spaces)
    s = re.sub(r'[^a-z0-9\s]', '', s)
    
    return s

def validate_email(email):
    """
    Validate email format using regex pattern.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email))) if pd.notna(email) else False

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]