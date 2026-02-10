import csv
import sys

def clean_csv(input_file, output_file):
    """
    Read a CSV file, remove rows that are completely empty,
    trim whitespace from all fields, and write to a new file.
    """
    cleaned_rows = []
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            for row in reader:
                # Check if row is completely empty
                if all(cell.strip() == '' for cell in row):
                    continue
                # Trim whitespace from each cell
                trimmed_row = [cell.strip() for cell in row]
                cleaned_rows.append(trimmed_row)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(cleaned_rows)
            
        print(f"Cleaned data written to {output_file}")
        print(f"Removed {len(cleaned_rows) - len(cleaned_rows)} empty rows")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_csv(input_file, output_file)