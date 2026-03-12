
import re

def clean_string(text):
    """
    Clean and normalize a string by:
    1. Stripping leading/trailing whitespace
    2. Replacing multiple spaces with a single space
    3. Converting to lowercase
    """
    if not isinstance(text, str):
        return text
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text.lower()