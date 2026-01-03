
import pandas as pd
import re

def clean_dataframe(df, text_column):
    """
    Remove duplicate rows and normalize text in specified column.
    """
    # Remove duplicates
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize text: lowercase, remove extra whitespace
    df_clean[text_column] = df_clean[text_column].apply(
        lambda x: re.sub(r'\s+', ' ', str(x).strip().lower())
    )
    
    return df_clean

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4],
        'text': ['Hello World', 'Python Code', 'python code', '  DATA  ', 'Test']
    })
    
    cleaned = clean_dataframe(sample_data, 'text')
    save_cleaned_data(cleaned, 'cleaned_data.csv')
    
    print("Original shape:", sample_data.shape)
    print("Cleaned shape:", cleaned.shape)
    print("\nCleaned data preview:")
    print(cleaned.head())