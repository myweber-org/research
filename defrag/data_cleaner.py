import csv
import re
from typing import List, Dict, Any, Optional

def clean_csv_data(input_file: str, output_file: str, columns_to_clean: Optional[List[str]] = None) -> None:
    """
    Clean data in a CSV file by removing extra whitespace and standardizing text.
    """
    cleaned_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            cleaned_row = {}
            for key, value in row.items():
                if columns_to_clean is None or key in columns_to_clean:
                    if isinstance(value, str):
                        # Remove extra whitespace
                        cleaned_value = re.sub(r'\s+', ' ', value.strip())
                        # Convert to title case for consistency
                        cleaned_value = cleaned_value.title()
                    else:
                        cleaned_value = value
                else:
                    cleaned_value = value
                cleaned_row[key] = cleaned_value
            cleaned_rows.append(cleaned_row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

def validate_email_format(email: str) -> bool:
    """
    Validate if a string is in a proper email format.
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))

def filter_invalid_emails(input_file: str, email_column: str) -> List[Dict[str, Any]]:
    """
    Filter rows from a CSV file where the email column contains invalid email addresses.
    """
    valid_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        
        for row in reader:
            if email_column in row and validate_email_format(row[email_column]):
                valid_rows.append(row)
    
    return valid_rows

def remove_duplicates(data: List[Dict[str, Any]], key_column: str) -> List[Dict[str, Any]]:
    """
    Remove duplicate rows based on a specified key column.
    """
    seen = set()
    unique_data = []
    
    for row in data:
        key_value = row.get(key_column)
        if key_value not in seen:
            seen.add(key_value)
            unique_data.append(row)
    
    return unique_data