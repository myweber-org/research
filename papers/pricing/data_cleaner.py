import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def filter_by_threshold(df, column, threshold):
    """
    Filter DataFrame rows where column value is greater than threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to apply threshold.
        threshold (float): Threshold value.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    filtered_df = df[df[column] > threshold]
    return filtered_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, None, 4, 2],
        'B': [5, 6, 7, None, 6],
        'C': [10, 20, 30, 40, 20]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataset(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    filtered_df = filter_by_threshold(cleaned_df, 'C', 15)
    print("Filtered DataFrame (C > 15):")
    print(filtered_df)