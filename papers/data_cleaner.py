
import re

def filter_valid_emails(email_list):
    """
    Filters a list of email strings, returning only those that match a basic
    email pattern.
    """
    if not isinstance(email_list, list):
        raise TypeError("Input must be a list")
    
    valid_emails = []
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    for email in email_list:
        if isinstance(email, str) and re.match(pattern, email):
            valid_emails.append(email)
    
    return valid_emails

def example_usage():
    """Example of how to use the filter_valid_emails function."""
    sample_emails = [
        "user@example.com",
        "invalid-email",
        "another.user@domain.org",
        "not.an.email@",
        "test@sub.domain.co.uk"
    ]
    
    result = filter_valid_emails(sample_emails)
    print("Original list:", sample_emails)
    print("Valid emails:", result)
    return result

if __name__ == "__main__":
    example_usage()