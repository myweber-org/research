
import pandas as pd

def clean_dataset(df, id_column='id'):
    """
    Remove duplicate rows based on ID column and standardize column names.
    """
    if df.empty:
        return df
    
    # Remove duplicates
    if id_column in df.columns:
        df = df.drop_duplicates(subset=[id_column], keep='first')
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    return df

def validate_dataframe(df, required_columns):
    """
    Validate that required columns exist in the dataframe.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        columns_to_check (list, optional): Specific columns to check for duplicates. 
                                          If None, checks all columns.
        fill_strategy (str): Strategy to fill missing values. 
                            Options: 'mean', 'median', 'mode', 'zero', 'drop'
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    else:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                elif fill_strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unsupported fill strategy: {fill_strategy}")
                
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
    
    # For categorical columns, fill with most frequent value
    for column in cleaned_df.select_dtypes(include=['object']).columns:
        if cleaned_df[column].isnull().any():
            most_frequent = cleaned_df[column].mode()[0]
            cleaned_df[column] = cleaned_df[column].fillna(most_frequent)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of columns that must be present
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, message)
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

# Example usage function
def process_sample_data():
    """Demonstrate the data cleaning functionality with sample data."""
    # Create sample data with duplicates and missing values
    sample_data = {
        'id': [1, 2, 3, 1, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Eve'],
        'age': [25, 30, np.nan, 25, 35],
        'score': [85, 90, 78, 85, np.nan],
        'department': ['HR', 'IT', 'IT', 'HR', 'Finance']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nShape:", df.shape)
    
    # Clean the data
    cleaned_df = clean_dataset(df, columns_to_check=['id'], fill_strategy='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nShape after cleaning:", cleaned_df.shape)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned_df, required_columns=['id', 'name', 'age'])
    print(f"\nValidation: {message}")
    
    return cleaned_df

if __name__ == "__main__":
    process_sample_data()