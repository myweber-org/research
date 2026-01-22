
import re

def clean_string(text):
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r'\s+', ' ', text.strip())
    return cleaned.lower()