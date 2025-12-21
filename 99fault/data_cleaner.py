
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_stats(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if data.empty:
        return {}
    
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column]),
        'count': len(data[column])
    }
    return stats
import pandas as pd
import re

def clean_dataframe(df, column_name):
    """
    Remove duplicate rows and normalize string values in specified column.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize strings: lowercase, remove extra whitespace
    if column_name in df_cleaned.columns:
        df_cleaned[column_name] = df_cleaned[column_name].apply(
            lambda x: re.sub(r'\s+', ' ', str(x).strip().lower()) if pd.notnull(x) else x
        )
    
    return df_cleaned

def validate_email_column(df, email_column):
    """
    Validate email format in specified column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['is_valid_email'] = df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    return df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Alice   Johnson  '],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'alice@company.net']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    df_cleaned = clean_dataframe(df, 'name')
    print("\nCleaned DataFrame:")
    print(df_cleaned)
    
    df_validated = validate_email_column(df_cleaned, 'email')
    print("\nDataFrame with email validation:")
    print(df_validated)