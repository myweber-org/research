
import pandas as pd
import numpy as np
import re

def clean_column_names(df):
    df.columns = [re.sub(r'[^a-zA-Z0-9]', '_', col).lower().strip() for col in df.columns]
    return df

def remove_duplicates(df, subset=None):
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if strategy == 'mean' and df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].mean(), inplace=True)
        elif strategy == 'median' and df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        elif strategy == 'mode':
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None, inplace=True)
        else:
            df[col].fillna('', inplace=True)
    return df

def standardize_text(df, columns):
    for col in columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.lower().str.strip()
    return df

def process_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        df = clean_column_names(df)
        df = remove_duplicates(df)
        df = fill_missing_values(df)
        
        text_columns = df.select_dtypes(include=['object']).columns
        df = standardize_text(df, text_columns)
        
        df.to_csv(output_file, index=False)
        print(f"Data cleaning completed. Cleaned data saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    process_csv(input_csv, output_csv)