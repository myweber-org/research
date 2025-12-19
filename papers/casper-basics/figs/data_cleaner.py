import csv
import re
from typing import List, Dict, Any

def clean_string(value: str) -> str:
    """Remove extra whitespace and convert to lowercase."""
    if not isinstance(value, str):
        return value
    return re.sub(r'\s+', ' ', value.strip()).lower()

def clean_numeric(value: str) -> float:
    """Convert string to float, handling empty values."""
    try:
        return float(value.replace(',', ''))
    except ValueError:
        return 0.0

def read_and_clean_csv(filepath: str) -> List[Dict[str, Any]]:
    """Read CSV file and apply cleaning functions to each row."""
    cleaned_data = []
    
    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            cleaned_row = {}
            for key, value in row.items():
                if key.endswith('_amount') or key.endswith('_price'):
                    cleaned_row[key] = clean_numeric(value)
                else:
                    cleaned_row[key] = clean_string(value)
            cleaned_data.append(cleaned_row)
    
    return cleaned_data

def write_cleaned_csv(data: List[Dict[str, Any]], output_path: str) -> None:
    """Write cleaned data to a new CSV file."""
    if not data:
        return
    
    fieldnames = data[0].keys()
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def remove_duplicates(data: List[Dict[str, Any]], key_field: str) -> List[Dict[str, Any]]:
    """Remove duplicate rows based on a specified key field."""
    seen = set()
    unique_data = []
    
    for row in data:
        identifier = row.get(key_field)
        if identifier not in seen:
            seen.add(identifier)
            unique_data.append(row)
    
    return unique_data

def filter_by_threshold(data: List[Dict[str, Any]], 
                        field: str, 
                        threshold: float) -> List[Dict[str, Any]]:
    """Filter rows where the specified field meets or exceeds threshold."""
    return [row for row in data if row.get(field, 0) >= threshold]