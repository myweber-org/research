import pandas as pd

def remove_duplicates(input_file, output_file, key_columns):
    """
    Load a CSV file, remove duplicate rows based on specified columns,
    and save the cleaned data to a new file.
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        df_cleaned = df.drop_duplicates(subset=key_columns, keep='first')
        final_count = len(df_cleaned)
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Data cleaning completed.")
        print(f"Initial records: {initial_count}")
        print(f"Final records: {final_count}")
        print(f"Duplicates removed: {initial_count - final_count}")
        print(f"Cleaned data saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    unique_keys = ["id", "email"]
    
    cleaned_data = remove_duplicates(input_csv, output_csv, unique_keys)