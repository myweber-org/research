import pandas as pd
import numpy as np
from datetime import datetime

def clean_dataset(input_file, output_file):
    """
    Load a dataset, remove duplicate rows, standardize date formats,
    and fill missing numeric values with column median.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove exact duplicates
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(df)
        
        # Standardize date columns (assuming columns with 'date' in name)
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Fill numeric missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        
        print(f"Cleaning complete:")
        print(f"  - Removed {duplicates_removed} duplicate rows")
        print(f"  - Standardized {len(date_columns)} date columns")
        print(f"  - Processed {len(numeric_cols)} numeric columns")
        print(f"  - Output saved to: {output_file}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    
    cleaned_data = clean_dataset(input_path, output_path)
    
    if cleaned_data is not None:
        print(f"Final dataset shape: {cleaned_data.shape}")