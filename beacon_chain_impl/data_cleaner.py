import re
from typing import List, Optional

def remove_special_characters(text: str, keep_spaces: bool = True) -> str:
    """
    Remove all non-alphanumeric characters from the input string.
    Optionally preserve spaces.
    """
    if keep_spaces:
        pattern = r'[^A-Za-z0-9\s]'
    else:
        pattern = r'[^A-Za-z0-9]'
    return re.sub(pattern, '', text)

def normalize_whitespace(text: str) -> str:
    """
    Replace multiple whitespace characters with a single space.
    Also strip leading and trailing whitespace.
    """
    return ' '.join(text.split())

def tokenize_text(text: str, lowercase: bool = True) -> List[str]:
    """
    Split text into tokens (words). Optionally convert to lowercase.
    """
    if lowercase:
        text = text.lower()
    tokens = text.split()
    return tokens

def clean_text_pipeline(
    text: str,
    remove_special: bool = True,
    normalize_ws: bool = True,
    tokenize: bool = False
) -> Optional[str]:
    """
    Apply a series of cleaning operations to the input text.
    Returns cleaned string or token list based on parameters.
    """
    if not isinstance(text, str):
        return None

    cleaned = text

    if remove_special:
        cleaned = remove_special_characters(cleaned)

    if normalize_ws:
        cleaned = normalize_whitespace(cleaned)

    if tokenize:
        return tokenize_text(cleaned)

    return cleaned

def batch_clean_texts(texts: List[str], **kwargs) -> List[Optional[str]]:
    """
    Apply cleaning pipeline to a list of text strings.
    Returns list of cleaned texts (or None for invalid inputs).
    """
    return [clean_text_pipeline(text, **kwargs) for text in texts]