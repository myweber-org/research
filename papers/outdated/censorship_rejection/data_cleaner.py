import csv
import re

def remove_duplicates(input_file, output_file):
    """Remove duplicate rows from a CSV file."""
    seen = set()
    unique_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            row_tuple = tuple(row)
            if row_tuple not in seen:
                seen.add(row_tuple)
                unique_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(unique_rows)
    
    return len(unique_rows)

def clean_numeric_columns(input_file, output_file, columns):
    """Clean numeric columns by removing non-numeric characters."""
    cleaned_data = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        cleaned_data.append(fieldnames)
        
        for row in reader:
            cleaned_row = []
            for field in fieldnames:
                value = row[field]
                if field in columns:
                    # Remove all non-numeric characters except decimal point
                    value = re.sub(r'[^\d.]', '', value)
                    # Handle empty results
                    if value == '':
                        value = '0'
                cleaned_row.append(value)
            cleaned_data.append(cleaned_row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(cleaned_data)
    
    return len(cleaned_data) - 1

def validate_email_format(email):
    """Validate email format using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def filter_valid_emails(input_file, output_file, email_column):
    """Filter rows with valid email addresses."""
    valid_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        valid_rows.append(fieldnames)
        
        for row in reader:
            if validate_email_format(row[email_column]):
                valid_rows.append([row[field] for field in fieldnames])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(valid_rows)
    
    return len(valid_rows) - 1

def merge_csv_files(file_list, output_file):
    """Merge multiple CSV files into one."""
    all_data = []
    headers_set = set()
    
    for file_path in file_list:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            headers_set.add(tuple(headers))
            
            if not all_data:
                all_data.append(headers)
            
            for row in reader:
                all_data.append(row)
    
    if len(headers_set) > 1:
        raise ValueError("CSV files have different headers")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(all_data)
    
    return len(all_data) - 1