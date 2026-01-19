import pandas as pd
import numpy as np
import logging

def clean_csv_data(file_path, output_path=None):
    """
    Clean a CSV file by handling missing values, removing duplicates,
    and standardizing string columns.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path} with shape {df.shape}")
        
        original_rows = df.shape[0]
        
        df = df.drop_duplicates()
        logging.info(f"Removed {original_rows - df.shape[0]} duplicate rows")
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip().str.lower()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            df[col] = df[col].fillna('unknown')
        
        if output_path:
            df.to_csv(output_path, index=False)
            logging.info(f"Cleaned data saved to {output_path}")
        
        return df
    
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error("The CSV file is empty")
        raise
    except Exception as e:
        logging.error(f"An error occurred during data cleaning: {str(e)}")
        raise

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and basic integrity checks.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    sample_data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', None],
        'age': [25, 30, None, 25, 35],
        'score': [85.5, 92.0, 78.5, 85.5, 88.0]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv')
    print(f"Cleaned data shape: {cleaned_df.shape}")
    print(cleaned_df.head())