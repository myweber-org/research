
import csv
import os
from typing import List, Dict, Any, Optional

def read_csv_file(file_path: str) -> List[Dict[str, Any]]:
    """Read a CSV file and return a list of dictionaries."""
    data = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")
    return data

def clean_numeric_field(record: Dict[str, Any], field: str, default: Any = 0) -> None:
    """Clean a numeric field in a record, converting to float if possible."""
    if field in record:
        try:
            record[field] = float(record[field])
        except (ValueError, TypeError):
            record[field] = default

def remove_empty_records(data: List[Dict[str, Any]], required_fields: List[str]) -> List[Dict[str, Any]]:
    """Remove records that have empty values in required fields."""
    cleaned_data = []
    for record in data:
        keep = True
        for field in required_fields:
            if field not in record or record[field] in (None, "", " "):
                keep = False
                break
        if keep:
            cleaned_data.append(record)
    return cleaned_data

def write_csv_file(data: List[Dict[str, Any]], file_path: str) -> bool:
    """Write data to a CSV file."""
    if not data:
        print("No data to write.")
        return False
    
    try:
        fieldnames = data[0].keys()
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

def clean_csv_data(input_path: str, output_path: str, numeric_fields: Optional[List[str]] = None, required_fields: Optional[List[str]] = None) -> None:
    """Main function to clean CSV data."""
    if numeric_fields is None:
        numeric_fields = []
    if required_fields is None:
        required_fields = []
    
    print(f"Reading data from {input_path}")
    data = read_csv_file(input_path)
    
    if not data:
        print("No data loaded.")
        return
    
    print(f"Loaded {len(data)} records.")
    
    for record in data:
        for field in numeric_fields:
            clean_numeric_field(record, field)
    
    cleaned_data = remove_empty_records(data, required_fields)
    print(f"After cleaning, {len(cleaned_data)} records remain.")
    
    if write_csv_file(cleaned_data, output_path):
        print(f"Cleaned data written to {output_path}")
    else:
        print("Failed to write cleaned data.")