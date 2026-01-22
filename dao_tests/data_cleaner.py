import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        file_path: Path to input CSV file
        output_path: Path for cleaned output CSV (optional)
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'drop', 'zero')
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original data shape: {df.shape}")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        print(f"Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                df[col].fillna(df[col].mean(), inplace=True)
        elif missing_strategy == 'median':
            for col in numeric_cols:
                df[col].fillna(df[col].median(), inplace=True)
        elif missing_strategy == 'zero':
            df.fillna(0, inplace=True)
        elif missing_strategy == 'drop':
            df.dropna(inplace=True)
        
        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
        
        print(f"Cleaned data shape: {df.shape}")
        print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
        
        # Save to file if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df: pandas.DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        print(f"Warning: Found {inf_count} infinite values in numeric columns")
    
    return True

# Example usage
if __name__ == "__main__":
    # This would be used for testing the module
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned = clean_csv_data('test_data.csv', 'cleaned_data.csv', 'mean')
    
    if cleaned is not None:
        is_valid = validate_dataframe(cleaned, ['A', 'B', 'C'])
        print(f"Data validation result: {is_valid}")