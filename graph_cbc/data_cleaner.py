
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