
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Load a CSV file, clean missing values, and save cleaned data.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove rows where critical columns are still null
        critical_columns = ['id', 'timestamp'] if 'id' in df.columns and 'timestamp' in df.columns else []
        for col in critical_columns:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaning complete. Cleaned data saved to {output_file}")
        print(f"Original rows: {len(pd.read_csv(input_file))}, Cleaned rows: {len(df)}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    clean_csv_data('raw_data.csv', 'cleaned_data.csv')