import csv
import re
from typing import List, Optional

def clean_string(value: str) -> str:
    """Remove extra whitespace and normalize string."""
    if not isinstance(value, str):
        return str(value) if value is not None else ""
    cleaned = re.sub(r'\s+', ' ', value.strip())
    return cleaned

def clean_numeric(value: str) -> Optional[float]:
    """Convert string to float, handling common issues."""
    if value is None:
        return None
    cleaned = clean_string(value)
    cleaned = cleaned.replace(',', '')
    try:
        return float(cleaned)
    except ValueError:
        return None

def read_csv_with_cleaning(filepath: str) -> List[dict]:
    """Read CSV file and clean all string fields."""
    cleaned_data = []
    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cleaned_row = {}
            for key, value in row.items():
                if key.endswith('_numeric') or 'amount' in key.lower():
                    cleaned_row[key] = clean_numeric(value)
                else:
                    cleaned_row[key] = clean_string(value)
            cleaned_data.append(cleaned_row)
    return cleaned_data

def write_cleaned_csv(data: List[dict], output_path: str) -> None:
    """Write cleaned data to a new CSV file."""
    if not data:
        return
    fieldnames = data[0].keys()
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def validate_email(email: str) -> bool:
    """Basic email validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, clean_string(email)))