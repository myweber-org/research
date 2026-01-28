import pandas as pd
import numpy as np

def clean_csv_data(filepath, strategy='mean', columns=None):
    """
    Clean a CSV file by handling missing values.
    
    Args:
        filepath (str): Path to the CSV file.
        strategy (str): Strategy for handling missing values. 
                       Options: 'mean', 'median', 'mode', 'drop'.
        columns (list): List of column names to apply cleaning to.
                       If None, applies to all numeric columns.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    df = pd.read_csv(filepath)
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    if strategy == 'drop':
        df_cleaned = df.dropna(subset=columns)
    else:
        df_cleaned = df.copy()
        for col in columns:
            if df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df[col].mean()
                elif strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'mode':
                    fill_value = df[col].mode()[0]
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                df_cleaned[col] = df[col].fillna(fill_value)
    
    return df_cleaned

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    
    Args:
        df (pandas.DataFrame): DataFrame to save.
        output_path (str): Path for output CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, strategy='median')
        save_cleaned_data(cleaned_df, output_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error during data cleaning: {e}")