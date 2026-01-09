
def remove_duplicates(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport csv
import re
from typing import List, Dict, Any, Optional

def clean_csv_data(input_file: str, output_file: str, columns_to_clean: Optional[List[str]] = None) -> None:
    """
    Clean data in a CSV file by removing extra whitespace and standardizing text.
    
    Args:
        input_file: Path to the input CSV file.
        output_file: Path to the output cleaned CSV file.
        columns_to_clean: List of column names to clean. If None, all columns are cleaned.
    """
    cleaned_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            cleaned_row = {}
            for field in fieldnames:
                value = row[field]
                
                if columns_to_clean is None or field in columns_to_clean:
                    if isinstance(value, str):
                        # Remove leading/trailing whitespace
                        value = value.strip()
                        # Replace multiple spaces with single space
                        value = re.sub(r'\s+', ' ', value)
                        # Standardize capitalization for certain fields
                        if field.lower() in ['name', 'city', 'country']:
                            value = value.title()
                
                cleaned_row[field] = value
            cleaned_rows.append(cleaned_row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

def validate_email_format(email: str) -> bool:
    """
    Validate if a string has a basic email format.
    
    Args:
        email: Email string to validate.
    
    Returns:
        True if the email has a valid format, False otherwise.
    """
    if not isinstance(email, str):
        return False
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))

def remove_duplicate_rows(data: List[Dict[str, Any]], key_columns: List[str]) -> List[Dict[str, Any]]:
    """
    Remove duplicate rows based on specified key columns.
    
    Args:
        data: List of dictionaries representing rows.
        key_columns: List of column names to use for duplicate detection.
    
    Returns:
        List of unique rows.
    """
    seen = set()
    unique_data = []
    
    for row in data:
        key = tuple(str(row.get(col, '')) for col in key_columns)
        if key not in seen:
            seen.add(key)
            unique_data.append(row)
    
    return unique_data

def standardize_phone_number(phone: str) -> str:
    """
    Standardize phone number format by removing non-digit characters.
    
    Args:
        phone: Phone number string to standardize.
    
    Returns:
        Standardized phone number with only digits.
    """
    if not isinstance(phone, str):
        return ''
    
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    return digits

if __name__ == "__main__":
    # Example usage
    sample_data = [
        {"name": "john doe  ", "email": "john@example.com", "phone": "(123) 456-7890"},
        {"name": "Jane Smith", "email": "jane@example.com", "phone": "987-654-3210"},
        {"name": "john doe", "email": "john@example.com", "phone": "1234567890"}
    ]
    
    # Write sample data to CSV
    with open('sample_input.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["name", "email", "phone"])
        writer.writeheader()
        writer.writerows(sample_data)
    
    # Clean the CSV data
    clean_csv_data('sample_input.csv', 'sample_output.csv', columns_to_clean=["name", "phone"])
    
    # Test email validation
    test_emails = ["test@example.com", "invalid-email", "another@test.co.uk"]
    for email in test_emails:
        print(f"{email}: {validate_email_format(email)}")
    
    # Test duplicate removal
    unique_data = remove_duplicate_rows(sample_data, ["email"])
    print(f"Original rows: {len(sample_data)}, Unique rows: {len(unique_data)}")
    
    # Test phone standardization
    test_phones = ["(123) 456-7890", "987-654-3210", "+1 800 555 1234"]
    for phone in test_phones:
        print(f"{phone} -> {standardize_phone_number(phone)}")