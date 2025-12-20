
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_duplicates = initial_rows - len(cleaned_df)
        print(f"Removed {removed_duplicates} duplicate rows.")
    
    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna('Unknown')
        
        print("Missing values have been filled.")
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate the DataFrame for common data quality issues.
    """
    issues = []
    
    if df.empty:
        issues.append("DataFrame is empty.")
    
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        issues.append(f"Found {missing_values} missing values.")
    
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        issues.append(f"Found {duplicate_rows} duplicate rows.")
    
    if issues:
        print("Data validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Data validation passed.")
        return True

def main():
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
        'age': [25, 30, None, 40, 35, 35, 28],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    validation_result = validate_dataframe(cleaned_df)
    
    if validation_result:
        print("Data is ready for analysis.")
    else:
        print("Data requires further cleaning.")

if __name__ == "__main__":
    main()