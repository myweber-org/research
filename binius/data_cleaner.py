
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Load a CSV file, clean missing values, and save cleaned data.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to save cleaned CSV file
    missing_strategy (str): Strategy for handling missing values
                           'mean' - fill with column mean
                           'median' - fill with column median
                           'drop' - drop rows with missing values
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        if missing_strategy == 'drop':
            df_cleaned = df.dropna()
            print(f"After dropping missing values: {df_cleaned.shape}")
        elif missing_strategy in ['mean', 'median']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if missing_strategy == 'mean':
                    fill_value = df[col].mean()
                else:
                    fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
            df_cleaned = df
            print(f"Filled missing values using {missing_strategy}")
        else:
            raise ValueError("Invalid missing_strategy. Use 'mean', 'median', or 'drop'")
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df, min_rows=10):
    """
    Validate that cleaned data meets minimum requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if data is valid, False otherwise
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: Less than {min_rows} rows")
        return False
    
    if df.isnull().any().any():
        print("Validation failed: DataFrame contains null values")
        return False
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_data = clean_csv_data(input_csv, output_csv, missing_strategy='mean')
    
    if cleaned_data is not None:
        is_valid = validate_data(cleaned_data)
        if is_valid:
            print("Data cleaning completed successfully")
        else:
            print("Data cleaning completed but validation failed")