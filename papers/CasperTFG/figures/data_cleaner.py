
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_list = [1, 2, 2, 3, 4, 3, 5, 1, 6]
    cleaned_list = remove_duplicates_preserve_order(sample_list)
    print(f"Original list: {sample_list}")
    print(f"Cleaned list: {cleaned_list}")
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None):
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        df_cleaned = df.drop_duplicates()
        final_count = len(df_cleaned)
        
        if output_file is None:
            output_file = input_file.replace('.csv', '_cleaned.csv')
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Removed {initial_count - final_count} duplicate rows")
        print(f"Original: {initial_count} rows, Cleaned: {final_count} rows")
        print(f"Cleaned data saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file.csv> [output_file.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    remove_duplicates(input_file, output_file)