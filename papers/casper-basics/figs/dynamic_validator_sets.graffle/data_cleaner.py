
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: Input pandas DataFrame
        column_mapping: Dictionary to rename columns (old_name: new_name)
        drop_duplicates: Boolean to remove duplicate rows
        normalize_text: Boolean to normalize text columns (strip, lower case)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    return cleaned_df

def remove_special_characters(text, keep_pattern=r'[a-zA-Z0-9\s]'):
    """
    Remove special characters from text, keeping only alphanumeric and spaces.
    
    Args:
        text: Input string
        keep_pattern: Regex pattern for characters to keep
    
    Returns:
        Cleaned string
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    return re.sub(f'[^{keep_pattern}]', '', text)

def validate_email(email):
    """
    Validate email format using regex pattern.
    
    Args:
        email: Email string to validate
    
    Returns:
        Boolean indicating if email is valid
    """
    if pd.isna(email):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def format_phone_number(phone):
    """
    Format phone number to standard format (XXX) XXX-XXXX.
    
    Args:
        phone: Phone number string
    
    Returns:
        Formatted phone number or original if invalid
    """
    if pd.isna(phone):
        return phone
    
    phone = str(phone)
    digits = re.sub(r'\D', '', phone)
    
    if len(digits) == 10:
        return f'({digits[:3]}) {digits[3:6]}-{digits[6:]}'
    elif len(digits) == 11 and digits[0] == '1':
        return f'({digits[1:4]}) {digits[4:7]}-{digits[7:]}'
    
    return phone

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', ' Bob Johnson '],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.net'],
        'phone': ['1234567890', '555-123-4567', '9876543210', '1-800-555-1234'],
        'notes': ['Important client!', 'Regular customer.', 'VIP #1', 'Needs follow-up.']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataframe(df, drop_duplicates=True, normalize_text=True)
    cleaned['phone'] = cleaned['phone'].apply(format_phone_number)
    cleaned['email_valid'] = cleaned['email'].apply(validate_email)
    
    print("Cleaned DataFrame:")
    print(cleaned)