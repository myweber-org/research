
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport re
from typing import List, Optional

def remove_special_characters(text: str, keep_spaces: bool = True) -> str:
    """
    Remove all non-alphanumeric characters from the input string.
    
    Args:
        text: The input string to clean.
        keep_spaces: If True, preserve spaces. If False, remove spaces as well.
    
    Returns:
        A cleaned string with only alphanumeric characters and optionally spaces.
    """
    if keep_spaces:
        pattern = r'[^A-Za-z0-9\s]'
    else:
        pattern = r'[^A-Za-z0-9]'
    
    return re.sub(pattern, '', text)

def normalize_whitespace(text: str) -> str:
    """
    Replace multiple consecutive whitespace characters with a single space.
    
    Args:
        text: The input string to normalize.
    
    Returns:
        A string with normalized whitespace.
    """
    return re.sub(r'\s+', ' ', text).strip()

def clean_text_pipeline(text: str, 
                       remove_special: bool = True, 
                       normalize_space: bool = True,
                       to_lower: bool = False) -> str:
    """
    Apply a series of cleaning operations to the input text.
    
    Args:
        text: The input string to process.
        remove_special: If True, remove special characters.
        normalize_space: If True, normalize whitespace.
        to_lower: If True, convert text to lowercase.
    
    Returns:
        A cleaned version of the input text.
    """
    result = text
    
    if remove_special:
        result = remove_special_characters(result)
    
    if normalize_space:
        result = normalize_whitespace(result)
    
    if to_lower:
        result = result.lower()
    
    return result

def batch_clean_texts(texts: List[str], **kwargs) -> List[str]:
    """
    Apply cleaning pipeline to a list of text strings.
    
    Args:
        texts: A list of text strings to clean.
        **kwargs: Additional arguments to pass to clean_text_pipeline.
    
    Returns:
        A list of cleaned text strings.
    """
    return [clean_text_pipeline(text, **kwargs) for text in texts]

def validate_email(email: str) -> bool:
    """
    Validate an email address format using regex.
    
    Args:
        email: The email address string to validate.
    
    Returns:
        True if the email format is valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def extract_digits(text: str) -> str:
    """
    Extract only digit characters from a string.
    
    Args:
        text: The input string containing digits.
    
    Returns:
        A string containing only the digits from the input.
    """
    return ''.join(re.findall(r'\d', text))