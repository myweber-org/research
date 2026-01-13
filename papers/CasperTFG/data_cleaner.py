import pandas as pd
import numpy as np

def clean_csv_data(input_path, output_path):
    """
    Load a CSV file, clean missing values, and save cleaned data.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Original shape: {df.shape}")
        
        # Handle missing values
        df_cleaned = df.copy()
        
        # Fill numeric columns with median
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            median_val = df_cleaned[col].median()
            df_cleaned[col].fillna(median_val, inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_val = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
            df_cleaned[col].fillna(mode_val, inplace=True)
        
        # Remove duplicate rows
        df_cleaned.drop_duplicates(inplace=True)
        
        # Reset index after cleaning
        df_cleaned.reset_index(drop=True, inplace=True)
        
        print(f"Cleaned shape: {df_cleaned.shape}")
        print(f"Missing values after cleaning: {df_cleaned.isnull().sum().sum()}")
        
        # Save cleaned data
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df):
    """
    Perform basic data validation checks.
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    checks_passed = True
    
    # Check for remaining missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Validation warning: {missing_count} missing values still present")
        checks_passed = False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            print(f"Validation warning: Infinite values found in column {col}")
            checks_passed = False
    
    # Check for negative values in columns that shouldn't have them
    positive_cols = ['age', 'salary', 'price']  # Example columns that should be positive
    for col in positive_cols:
        if col in df.columns:
            if (df[col] < 0).any():
                print(f"Validation warning: Negative values found in column {col}")
                checks_passed = False
    
    if checks_passed:
        print("All validation checks passed")
    
    return checks_passed

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        validate_data(cleaned_df)