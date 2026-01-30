
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str, optional): Path to save the cleaned CSV file.
                                     If None, overwrites the input file.
        subset (list, optional): Column labels to consider for identifying duplicates.
        keep (str, optional): Which duplicates to keep. Options: 'first', 'last', False.
                              'first' keeps the first occurrence.
                              'last' keeps the last occurrence.
                              False drops all duplicates.
    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        final_rows = len(df_cleaned)
        
        if output_file is None:
            output_file = input_file
        
        df_cleaned.to_csv(output_file, index=False)
        
        duplicates_removed = initial_rows - final_rows
        print(f"Cleaning complete. Removed {duplicates_removed} duplicate rows.")
        print(f"Original rows: {initial_rows}, Cleaned rows: {final_rows}")
        print(f"Saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)