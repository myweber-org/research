
import pandas as pd

def clean_dataframe(df):
    """
    Remove rows with any null values and standardize column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = (
        df_cleaned.columns
        .str.lower()
        .str.replace(' ', '_')
    )
    
    return df_cleaned