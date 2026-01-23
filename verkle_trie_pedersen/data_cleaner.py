
import re

def clean_email(email):
    """
    Clean and validate an email address.
    Returns a cleaned email string if valid, otherwise returns None.
    """
    if not isinstance(email, str):
        return None

    email = email.strip().lower()

    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return email
    else:
        return None

def clean_email_list(email_list):
    """
    Clean a list of email addresses.
    Returns a list of valid, cleaned email addresses.
    """
    if not isinstance(email_list, list):
        return []

    cleaned_emails = []
    for email in email_list:
        cleaned = clean_email(email)
        if cleaned:
            cleaned_emails.append(cleaned)

    return cleaned_emails