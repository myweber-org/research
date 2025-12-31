
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list: A list containing elements (must be hashable).
    
    Returns:
        A new list with duplicates removed.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import re

def clean_text(text):
    """
    Clean and normalize the input text by:
    - Removing leading and trailing whitespace.
    - Replacing multiple spaces/newlines/tabs with a single space.
    - Converting the text to lowercase.
    
    Args:
        text (str): The input text to be cleaned.
    
    Returns:
        str: The cleaned and normalized text.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Replace multiple whitespace characters (spaces, newlines, tabs) with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text