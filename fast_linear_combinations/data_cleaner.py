
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        initial_rows = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - df_clean.shape[0]
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                print(f"Filled missing values in column '{col}' with median")
        
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna('Unknown')
                print(f"Filled missing values in column '{col}' with 'Unknown'")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        print("Warning: DataFrame is empty")
    
    return True

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, np.nan, 15.0, 20.0, np.nan, 30.0],
        'category': ['A', 'B', 'B', None, 'C', 'A']
    })
    
    print("Original DataFrame:")
    print(sample_data)
    print("\nCleaning data...")
    
    cleaned_df = clean_dataframe(sample_data)
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    if validate_dataframe(cleaned_df, required_columns=['id', 'value']):
        print("\nData validation passed")
    
    save_cleaned_data(cleaned_df, "cleaned_data.csv")