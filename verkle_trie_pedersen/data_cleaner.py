
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
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

def main():
    sample_data = pd.DataFrame({
        'age': [25, 30, 35, 200, 45, 50, 55, 60, 65, 70],
        'salary': [50000, 55000, 60000, 1000000, 70000, 75000, 80000, 85000, 90000, 95000]
    })
    
    print("Original Data:")
    print(sample_data)
    
    cleaned_data = clean_dataset(sample_data, ['age', 'salary'])
    
    print("\nCleaned Data:")
    print(cleaned_data)
    
    return cleaned_data

if __name__ == "__main__":
    main()