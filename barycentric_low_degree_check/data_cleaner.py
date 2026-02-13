import pandas as pd
import numpy as np

def clean_csv_data(filepath, output_path=None):
    """
    Load a CSV file, clean missing values, and convert data types.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        print(f"Removed {duplicates_removed} duplicate rows")
        print(f"Final data shape: {df.shape}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    # Check for remaining null values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("Warning: DataFrame still contains null values")
        for col, count in null_counts[null_counts > 0].items():
            print(f"  {col}: {count} nulls")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, np.nan, 15.2, 20.1, np.nan],
        'category': ['A', 'B', None, 'A', 'C'],
        'date': ['2023-01-01', '2023-01-02', None, '2023-01-03', '2023-01-04']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
        print(f"Data validation result: {is_valid}")
        print("\nCleaned data preview:")
        print(cleaned_df.head())