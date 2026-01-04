
import pandas as pd
import re

def clean_dataframe(df, column_name):
    """
    Clean a specific column in a DataFrame by removing duplicates,
    stripping whitespace, and converting to lowercase.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].str.lower()
    df = df.drop_duplicates(subset=[column_name], keep='first')
    
    return df

def normalize_string(text):
    """
    Normalize a string by removing special characters and extra spaces.
    """
    if not isinstance(text, str):
        return text
    
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def process_csv(input_path, output_path, column_to_clean):
    """
    Read a CSV file, clean the specified column, and save the result.
    """
    df = pd.read_csv(input_path)
    df = clean_dataframe(df, column_to_clean)
    df[column_to_clean] = df[column_to_clean].apply(normalize_string)
    df.to_csv(output_path, index=False)
    
    return df

if __name__ == "__main__":
    input_file = "input_data.csv"
    output_file = "cleaned_data.csv"
    target_column = "product_name"
    
    try:
        result_df = process_csv(input_file, output_file, target_column)
        print(f"Data cleaned successfully. Rows processed: {len(result_df)}")
    except Exception as e:
        print(f"Error during processing: {e}")