import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        file_path: Path to input CSV file
        output_path: Path for cleaned CSV file (optional)
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'drop')
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(file_path)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        df_cleaned = df.copy()
        
        if missing_strategy == 'drop':
            df_cleaned = df_cleaned.dropna()
        elif missing_strategy in ['mean', 'median']:
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if missing_strategy == 'mean':
                    fill_value = df_cleaned[col].mean()
                else:
                    fill_value = df_cleaned[col].median()
                
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
        
        df_cleaned = df_cleaned.drop_duplicates()
        
        print(f"Cleaned data shape: {df_cleaned.shape}")
        print(f"Remaining missing values: {df_cleaned.isnull().sum().sum()}")
        
        if output_path:
            df_cleaned.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df_cleaned
        
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
        bool: True if validation passes
    """
    if df is None or df.empty:
        print("Error: Dataframe is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    if df.isnull().sum().sum() > 0:
        print("Warning: Dataframe contains missing values")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, 7, 8, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv', 'mean')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, ['A', 'B', 'C'])
        print(f"Data validation result: {is_valid}")