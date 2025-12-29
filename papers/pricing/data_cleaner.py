
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Load a CSV file, clean missing values, and save cleaned data.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Original shape: {df.shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
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
        
        # Remove rows where all values are NaN
        df = df.dropna(how='all')
        
        print(f"Final cleaned shape: {df.shape}")
        print(f"Missing values after cleaning:\n{df.isnull().sum()}")
        
        # Save cleaned data
        if output_path is None:
            output_path = Path(input_path).stem + '_cleaned.csv'
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    print(f"DataFrame validation passed. Shape: {df.shape}")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Frank'],
        'age': [25, 30, None, 35, None, 40],
        'score': [85.5, 92.0, 78.5, None, 88.0, 95.5]
    }
    
    # Create sample DataFrame
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_data.csv', index=False)
    
    # Clean the sample data
    cleaned_df = clean_csv_data('sample_data.csv', 'sample_data_cleaned.csv')
    
    # Validate the cleaned data
    if cleaned_df is not None:
        validate_dataframe(cleaned_df, ['id', 'name', 'age', 'score'])