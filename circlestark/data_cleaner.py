
import re
import string

def normalize_text(text):
    """
    Normalize text by converting to lowercase, removing extra whitespace,
    and stripping punctuation from the edges.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Strip punctuation from the beginning and end
    text = text.strip(string.punctuation)
    
    return text

def remove_special_characters(text, keep_spaces=True):
    """
    Remove all non-alphanumeric characters from text.
    
    Args:
        text: Input string to clean
        keep_spaces: If True, preserve spaces between words
    
    Returns:
        Cleaned string containing only alphanumeric characters and optionally spaces
    """
    if not isinstance(text, str):
        return ""
    
    if keep_spaces:
        # Keep letters, numbers, and spaces
        pattern = r'[^a-zA-Z0-9\s]'
    else:
        # Keep only letters and numbers
        pattern = r'[^a-zA-Z0-9]'
    
    return re.sub(pattern, '', text)

def clean_whitespace(text):
    """
    Clean and normalize all whitespace in text.
    Replaces tabs, newlines, and multiple spaces with single spaces.
    """
    if not isinstance(text, str):
        return ""
    
    # Replace all whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def get_word_count(text):
    """
    Count the number of words in a text string.
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    
    # Split by whitespace and count non-empty elements
    words = [word for word in text.split() if word]
    return len(words)

def truncate_text(text, max_length, suffix="..."):
    """
    Truncate text to a maximum length, adding suffix if truncated.
    
    Args:
        text: Input text to truncate
        max_length: Maximum allowed length
        suffix: String to append if text is truncated
    
    Returns:
        Truncated text with suffix if necessary
    """
    if not isinstance(text, str):
        return ""
    
    if len(text) <= max_length:
        return text
    
    # Calculate truncation point accounting for suffix length
    truncate_point = max_length - len(suffix)
    if truncate_point <= 0:
        return suffix
    
    return text[:truncate_point].rstrip() + suffix