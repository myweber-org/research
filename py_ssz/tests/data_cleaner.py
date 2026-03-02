
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else '', inplace=True)
            else:
                cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0, inplace=True)
    
    return cleaned_dfimport pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        file_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output CSV. 
                                   If None, returns DataFrame
        missing_strategy (str): Strategy for handling missing values.
                              Options: 'mean', 'median', 'drop', 'zero'
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None
    """
    try:
        df = pd.read_csv(file_path)
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif missing_strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif missing_strategy == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        elif missing_strategy == 'drop':
            df = df.dropna(subset=numeric_cols)
        
        # Fill non-numeric columns with placeholder
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        print(f"Data cleaning completed:")
        print(f"  - Removed {duplicates_removed} duplicate rows")
        print(f"  - Final dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"  - Missing values handled with '{missing_strategy}' strategy")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"  - Cleaned data saved to: {output_path}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
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
    if not numeric_cols.empty:
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        if inf_count > 0:
            print(f"Validation warning: Found {inf_count} infinite values")
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, np.nan, 15.2, np.nan, 20.1, 20.1],
        'category': ['A', 'B', np.nan, 'A', 'C', 'C'],
        'score': [85, 92, np.nan, 78, 95, 95]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    # Clean the test data
    cleaned_df = clean_csv_data('test_data.csv', missing_strategy='mean')
    
    if cleaned_df is not None:
        # Validate the cleaned data
        validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
        
        # Display cleaned data
        print("\nCleaned DataFrame:")
        print(cleaned_df)
    
    # Clean up test file
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')