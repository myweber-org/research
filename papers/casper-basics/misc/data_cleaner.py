import csv
import re
from typing import List, Dict, Any

def remove_duplicates(data: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = set()
    unique_data = []
    for row in data:
        if row[key] not in seen:
            seen.add(row[key])
            unique_data.append(row)
    return unique_data

def normalize_string(value: str) -> str:
    if not isinstance(value, str):
        return value
    value = value.strip().lower()
    value = re.sub(r'\s+', ' ', value)
    return value

def clean_numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r'[^\d.-]', '', value)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0

def read_csv_file(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

def write_csv_file(data: List[Dict[str, Any]], filepath: str, fieldnames: List[str]) -> None:
    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def clean_csv_data(input_file: str, output_file: str, unique_key: str) -> None:
    data = read_csv_file(input_file)
    
    for row in data:
        for key, value in row.items():
            if isinstance(value, str):
                row[key] = normalize_string(value)
            elif key.endswith('_amount') or key.endswith('_price'):
                row[key] = clean_numeric(value)
    
    cleaned_data = remove_duplicates(data, unique_key)
    
    if cleaned_data:
        fieldnames = list(cleaned_data[0].keys())
        write_csv_file(cleaned_data, output_file, fieldnames)
        print(f"Cleaned data saved to {output_file}")
    else:
        print("No data to save")

if __name__ == "__main__":
    clean_csv_data("raw_data.csv", "cleaned_data.csv", "id")