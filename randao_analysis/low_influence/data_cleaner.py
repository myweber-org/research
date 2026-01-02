import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Load a CSV file, clean missing values, convert data types,
    and save the cleaned data to a new file.
    """
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        for col in categorical_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna('Unknown')
        
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    print(f"Converted column '{col}' to datetime")
                except (ValueError, TypeError):
                    pass
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        print(f"Final data shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty.")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_csv, output_csv)
    
    if cleaned_df is not None:
        print("Data cleaning completed successfully.")
        print(cleaned_df.head())