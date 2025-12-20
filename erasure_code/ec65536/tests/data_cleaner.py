
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import re

def clean_text_column(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.lower()
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    return df

def remove_duplicates(df, subset=None, keep='first'):
    return df.drop_duplicates(subset=subset, keep=keep)

def standardize_phone_numbers(df, column_name):
    def format_phone(phone):
        phone = re.sub(r'\D', '', str(phone))
        if len(phone) == 10:
            return f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
        return phone
    
    df[column_name] = df[column_name].apply(format_phone)
    return df

def clean_dataset(df, text_columns=None, phone_columns=None, deduplicate=True):
    if text_columns:
        for col in text_columns:
            df = clean_text_column(df, col)
    
    if phone_columns:
        for col in phone_columns:
            df = standardize_phone_numbers(df, col)
    
    if deduplicate:
        df = remove_duplicates(df)
    
    return df