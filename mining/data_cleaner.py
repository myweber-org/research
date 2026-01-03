import re
import unicodedata

def clean_text(text, remove_digits=False, keep_case=False):
    """
    Clean and normalize a given text string.

    Args:
        text (str): Input text to clean.
        remove_digits (bool): If True, remove all digits.
        keep_case (bool): If True, preserve original case.

    Returns:
        str: Cleaned text string.
    """
    if not isinstance(text, str):
        return ''

    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Optionally remove digits
    if remove_digits:
        text = re.sub(r'\d+', '', text)

    # Optionally convert to lowercase
    if not keep_case:
        text = text.lower()

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)

    return text.strip()

def tokenize_text(text, token_pattern=r'\b\w+\b'):
    """
    Tokenize text using a regular expression pattern.

    Args:
        text (str): Input text to tokenize.
        token_pattern (str): Regex pattern for tokenization.

    Returns:
        list: List of tokens.
    """
    cleaned = clean_text(text)
    tokens = re.findall(token_pattern, cleaned)
    return tokens

if __name__ == '__main__':
    sample = "Hello World! 123 This is a TEST.   "
    print(f"Original: '{sample}'")
    print(f"Cleaned: '{clean_text(sample)}'")
    print(f"Cleaned (no digits): '{clean_text(sample, remove_digits=True)}'")
    print(f"Tokens: {tokenize_text(sample)}")