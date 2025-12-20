import csv
import sys

def clean_csv(input_file, output_file):
    """
    Remove rows with missing values and trim whitespace from CSV.
    """
    cleaned_rows = []
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            cleaned_rows.append(headers)
            
            for row in reader:
                # Skip rows with empty cells
                if any(cell.strip() == '' for cell in row):
                    continue
                # Trim whitespace from all cells
                trimmed_row = [cell.strip() for cell in row]
                cleaned_rows.append(trimmed_row)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(cleaned_rows)
            
        print(f"Cleaned data saved to {output_file}")
        print(f"Removed {len(cleaned_rows) - 1} valid rows from original")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    clean_csv(sys.argv[1], sys.argv[2])