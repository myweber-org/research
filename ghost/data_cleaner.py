import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_method='mean', drop_threshold=0.5):
    """
    Load and clean a CSV file by handling missing values.
    
    Parameters:
    filepath (str): Path to the CSV file.
    fill_method (str): Method to fill missing values ('mean', 'median', 'mode', or 'zero').
    drop_threshold (float): Drop columns with missing values ratio above this threshold.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Calculate missing ratio per column
    missing_ratio = df.isnull().sum() / len(df)
    
    # Drop columns with high missing ratio
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    # Fill missing values based on specified method
    for column in df.columns:
        if df[column].isnull().any():
            if fill_method == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                fill_value = df[column].mean()
            elif fill_method == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                fill_value = df[column].median()
            elif fill_method == 'mode':
                fill_value = df[column].mode()[0] if not df[column].mode().empty else np.nan
            elif fill_method == 'zero':
                fill_value = 0
            else:
                fill_value = df[column].ffill().bfill()  # Fallback to forward/backward fill
            
            df[column] = df[column].fillna(fill_value)
    
    # Remove any remaining rows with missing values
    df = df.dropna()
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    return df

def export_cleaned_data(df, output_path):
    """
    Export cleaned DataFrame to CSV.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame.
    output_path (str): Path to save the cleaned CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, fill_method='median', drop_threshold=0.3)
        export_cleaned_data(cleaned_df, output_file)
        print(f"Original shape: {pd.read_csv(input_file).shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")
    except Exception as e:
        print(f"Error during data cleaning: {e}")