
import pandas as pd

def clean_dataframe(df):
    """
    Remove duplicate rows and fill missing numeric values with column median.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values with median
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    return df_cleaned

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 10, 40, 50],
        'C': ['x', 'y', 'x', 'z', 'y']
    })
    
    cleaned_df = clean_dataframe(sample_data)
    print("Original DataFrame:")
    print(sample_data)
    print("\nCleaned DataFrame:")
    print(cleaned_df)