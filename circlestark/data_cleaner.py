import re
import json
from datetime import datetime

def clean_string(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def validate_email(email):
    if not isinstance(email, str):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def parse_date(date_str, fmt='%Y-%m-%d'):
    try:
        return datetime.strptime(date_str, fmt).date()
    except (ValueError, TypeError):
        return None

def clean_json_data(raw_data):
    if not raw_data:
        return {}
    if isinstance(raw_data, str):
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            return {}
    elif isinstance(raw_data, dict):
        data = raw_data
    else:
        return {}
    
    cleaned = {}
    for key, value in data.items():
        if isinstance(value, str):
            cleaned[key] = clean_string(value)
        elif isinstance(value, (int, float, bool)):
            cleaned[key] = value
        elif value is None:
            cleaned[key] = None
        else:
            cleaned[key] = str(value)
    return cleaned

def filter_dict_by_keys(original_dict, allowed_keys):
    if not isinstance(original_dict, dict) or not isinstance(allowed_keys, list):
        return {}
    return {k: v for k, v in original_dict.items() if k in allowed_keys}