import csv
import re

def clean_csv(input_file, output_file, remove_duplicates=True, strip_whitespace=True):
    """
    Clean a CSV file by removing duplicates and stripping whitespace.
    """
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            rows = list(reader)
        
        if strip_whitespace:
            rows = [[cell.strip() for cell in row] for row in rows]
        
        if remove_duplicates:
            seen = set()
            unique_rows = []
            for row in rows:
                row_tuple = tuple(row)
                if row_tuple not in seen:
                    seen.add(row_tuple)
                    unique_rows.append(row)
            rows = unique_rows
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            writer.writerows(rows)
        
        return True, f"Cleaned data saved to {output_file}"
    
    except FileNotFoundError:
        return False, f"Input file {input_file} not found"
    except Exception as e:
        return False, f"An error occurred: {str(e)}"

def validate_email(email):
    """
    Validate email format using regex.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def filter_by_column(input_file, output_file, column_index, filter_value):
    """
    Filter rows based on a specific column value.
    """
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            filtered_rows = [row for row in reader if row[column_index] == filter_value]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            writer.writerows(filtered_rows)
        
        return True, f"Filtered data saved to {output_file}"
    
    except FileNotFoundError:
        return False, f"Input file {input_file} not found"
    except IndexError:
        return False, f"Column index {column_index} is out of range"
    except Exception as e:
        return False, f"An error occurred: {str(e)}"