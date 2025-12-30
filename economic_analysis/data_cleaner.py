
import pandas as pd

def clean_dataframe(df):
    """
    Remove rows with null values and standardize column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')
    
    return df_cleaned

def filter_by_threshold(df, column, threshold):
    """
    Filter DataFrame rows where column value exceeds threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to filter by.
        threshold (float): Threshold value.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    return df[df[column] > threshold]

def main():
    # Example usage
    data = {
        'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, None, 35, 40],
        'Score': [85.5, 92.0, 78.5, 88.0]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = clean_dataframe(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print()
    
    filtered_df = filter_by_threshold(cleaned_df, 'score', 80.0)
    print("Filtered DataFrame (score > 80):")
    print(filtered_df)

if __name__ == "__main__":
    main()