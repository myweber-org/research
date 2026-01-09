
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str, optional): Path to save the cleaned CSV file.
                                     If None, overwrites the input file.
        subset (list, optional): List of column names to consider for duplicates.
    
    Returns:
        int: Number of duplicates removed.
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        if subset:
            df_cleaned = df.drop_duplicates(subset=subset, keep='first')
        else:
            df_cleaned = df.drop_duplicates(keep='first')
        
        final_count = len(df_cleaned)
        duplicates_removed = initial_count - final_count
        
        if output_file is None:
            output_file = input_file
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Processed: {input_file}")
        print(f"Initial rows: {initial_count}")
        print(f"Final rows: {final_count}")
        print(f"Duplicates removed: {duplicates_removed}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return -1
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        return -1
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return -1

def main():
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)

if __name__ == "__main__":
    main()