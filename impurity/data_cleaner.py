import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def validate_dataframe(df):
    """
    Basic validation of DataFrame structure.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    if df.empty:
        return True
    return True

def clean_dataset(file_path, output_path=None):
    """
    Load, clean, and save a dataset from CSV file.
    
    Args:
        file_path (str): Path to input CSV file.
        output_path (str, optional): Path for cleaned output.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    
    if not validate_dataframe(df):
        print("Error: Invalid DataFrame structure")
        return None
    
    initial_count = len(df)
    cleaned_df = remove_duplicates(df)
    final_count = len(cleaned_df)
    
    duplicates_removed = initial_count - final_count
    print(f"Removed {duplicates_removed} duplicate rows")
    
    if output_path:
        cleaned_df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4, 4, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'David'],
        'value': [10, 20, 20, 30, 40, 40, 50]
    })
    
    print("Original DataFrame:")
    print(sample_data)
    print("\nCleaned DataFrame:")
    cleaned = remove_duplicates(sample_data, subset=['id', 'name'])
    print(cleaned)