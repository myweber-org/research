
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd

def clean_dataframe(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df.columns = [col.strip().lower().replace(' ', '_') for col in cleaned_df.columns]
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate the DataFrame structure and required columns.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Reads a CSV file, removes duplicate rows, and saves the cleaned data.
    """
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    initial_count = len(df)
    df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
    final_count = len(df_cleaned)
    duplicates_removed = initial_count - final_count

    if output_file is None:
        output_file = input_file.replace('.csv', '_cleaned.csv')

    df_cleaned.to_csv(output_file, index=False)

    print(f"Initial rows: {initial_count}")
    print(f"Final rows: {final_count}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Cleaned data saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_csv_file> [output_csv_file]")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    remove_duplicates(input_csv, output_csv)