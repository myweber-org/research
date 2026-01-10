import csv
import re

def clean_csv(input_file, output_file, columns_to_clean=None):
    """
    Clean a CSV file by removing extra whitespace and standardizing text.
    """
    cleaned_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            cleaned_row = {}
            for field in fieldnames:
                value = row[field]
                if value and isinstance(value, str):
                    # Remove extra whitespace
                    value = re.sub(r'\s+', ' ', value.strip())
                    # Standardize case for certain fields
                    if field.lower() in ['email', 'username']:
                        value = value.lower()
                cleaned_row[field] = value
            
            # Only clean specific columns if provided
            if columns_to_clean:
                for col in columns_to_clean:
                    if col in cleaned_row and cleaned_row[col]:
                        cleaned_row[col] = re.sub(r'[^\w\s]', '', cleaned_row[col])
            
            cleaned_rows.append(cleaned_row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    return len(cleaned_rows)

def validate_email(email):
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email)) if email else False

if __name__ == "__main__":
    # Example usage
    records_processed = clean_csv('input.csv', 'output.csv', ['name', 'address'])
    print(f"Processed {records_processed} records")