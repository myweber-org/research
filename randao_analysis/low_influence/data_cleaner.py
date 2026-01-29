import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    filepath (str): Path to the CSV file.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'zero').
    drop_threshold (float): Drop columns with missing values above this ratio (0.0 to 1.0).
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    # Calculate missing ratio per column
    missing_ratio = df.isnull().sum() / len(df)
    
    # Drop columns with missing ratio above threshold
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    if len(columns_to_drop) > 0:
        print(f"Dropped columns with >{drop_threshold*100}% missing values: {list(columns_to_drop)}")
    
    # Fill remaining missing values based on strategy
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fill_strategy == 'mean':
        fill_values = df[numeric_cols].mean()
    elif fill_strategy == 'median':
        fill_values = df[numeric_cols].median()
    elif fill_strategy == 'mode':
        fill_values = df[numeric_cols].mode().iloc[0]
    elif fill_strategy == 'zero':
        fill_values = 0
    else:
        print(f"Warning: Unknown fill strategy '{fill_strategy}'. Using 'mean'.")
        fill_values = df[numeric_cols].mean()
    
    # Apply filling to numeric columns
    df[numeric_cols] = df[numeric_cols].fillna(fill_values)
    
    # For non-numeric columns, fill with most frequent value
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        if df[col].isnull().any():
            most_frequent = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(most_frequent)
    
    # Remove any remaining rows with missing values (should be none after filling)
    df = df.dropna()
    
    final_shape = df.shape
    print(f"Cleaned data shape: {final_shape}")
    print(f"Removed {original_shape[0] - final_shape[0]} rows and {original_shape[1] - final_shape[1]} columns")
    
    return df

def export_cleaned_data(df, output_path):
    """
    Export cleaned DataFrame to CSV.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame.
    output_path (str): Path for output CSV file.
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data exported to: {output_path}")
        return True
    return False

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, fill_strategy='median', drop_threshold=0.3)
    
    if cleaned_df is not None:
        export_cleaned_data(cleaned_df, output_file)