import csv
import os

def clean_csv(input_file, output_file, remove_empty=True, delimiter=','):
    """
    Clean a CSV file by removing empty rows and stripping whitespace.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output cleaned CSV file.
        remove_empty (bool): Whether to remove rows that are completely empty.
        delimiter (str): Delimiter used in the CSV file.
    
    Returns:
        int: Number of rows written to the output file.
    """
    cleaned_rows = []
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter=delimiter)
            
            for row in reader:
                # Strip whitespace from each cell
                cleaned_row = [cell.strip() for cell in row]
                
                # Skip empty rows if remove_empty is True
                if remove_empty and all(cell == '' for cell in cleaned_row):
                    continue
                
                cleaned_rows.append(cleaned_row)
        
        # Write cleaned data to output file
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, delimiter=delimiter)
            writer.writerows(cleaned_rows)
        
        return len(cleaned_rows)
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return 0
    except Exception as e:
        print(f"Error cleaning CSV: {str(e)}")
        return 0

def validate_csv(file_path, required_columns=None, delimiter=','):
    """
    Validate a CSV file structure and content.
    
    Args:
        file_path (str): Path to the CSV file to validate.
        required_columns (list): List of column names that must be present.
        delimiter (str): Delimiter used in the CSV file.
    
    Returns:
        tuple: (is_valid, error_message, row_count, column_count)
    """
    if not os.path.exists(file_path):
        return False, f"File '{file_path}' does not exist.", 0, 0
    
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=delimiter)
            
            # Read header
            try:
                header = next(reader)
            except StopIteration:
                return False, "CSV file is empty.", 0, 0
            
            # Check required columns
            if required_columns:
                missing_columns = [col for col in required_columns if col not in header]
                if missing_columns:
                    return False, f"Missing required columns: {missing_columns}", 0, len(header)
            
            # Count rows
            row_count = 1  # Header row
            for row in reader:
                row_count += 1
            
            return True, "CSV file is valid.", row_count, len(header)
            
    except Exception as e:
        return False, f"Error reading CSV: {str(e)}", 0, 0

def merge_csv_files(file_list, output_file, delimiter=',', skip_duplicate_headers=True):
    """
    Merge multiple CSV files into a single file.
    
    Args:
        file_list (list): List of CSV file paths to merge.
        output_file (str): Path to the output merged CSV file.
        delimiter (str): Delimiter used in the CSV files.
        skip_duplicate_headers (bool): Whether to skip headers after the first file.
    
    Returns:
        int: Total number of rows in the merged file.
    """
    total_rows = 0
    first_file = True
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, delimiter=delimiter)
            
            for file_path in file_list:
                if not os.path.exists(file_path):
                    print(f"Warning: File '{file_path}' not found, skipping.")
                    continue
                
                with open(file_path, 'r', newline='', encoding='utf-8') as infile:
                    reader = csv.reader(infile, delimiter=delimiter)
                    
                    try:
                        header = next(reader)
                        
                        # Write header only for first file
                        if first_file:
                            writer.writerow(header)
                            total_rows += 1
                            first_file = False
                        
                        # Write all rows from current file
                        for row in reader:
                            writer.writerow(row)
                            total_rows += 1
                            
                    except StopIteration:
                        print(f"Warning: File '{file_path}' is empty, skipping.")
                        continue
        
        return total_rows
        
    except Exception as e:
        print(f"Error merging CSV files: {str(e)}")
        return 0

if __name__ == "__main__":
    # Example usage
    input_csv = "data/raw_data.csv"
    cleaned_csv = "data/cleaned_data.csv"
    
    if os.path.exists(input_csv):
        rows_written = clean_csv(input_csv, cleaned_csv)
        print(f"Cleaned {rows_written} rows from '{input_csv}' to '{cleaned_csv}'")
        
        is_valid, message, rows, cols = validate_csv(cleaned_csv)
        print(f"Validation: {message}")
        print(f"Rows: {rows}, Columns: {cols}")
    else:
        print(f"Input file '{input_csv}' not found. Skipping example.")