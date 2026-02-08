
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    missing_strategy (str): Strategy for handling missing values.
        Options: 'mean', 'median', 'mode', 'drop'. Default is 'mean'.
    outlier_method (str): Method for detecting and handling outliers.
        Options: 'iqr', 'zscore', 'remove', 'cap'. Default is 'iqr'.
    columns (list): List of column names to clean. If None, clean all numeric columns.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df_clean.columns:
            continue
        
        if missing_strategy != 'drop':
            if missing_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif missing_strategy == 'median':
                fill_value = df_clean[col].median()
            elif missing_strategy == 'mode':
                fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else np.nan
            else:
                fill_value = np.nan
            
            df_clean[col].fillna(fill_value, inplace=True)
        else:
            df_clean.dropna(subset=[col], inplace=True)
        
        if outlier_method in ['iqr', 'zscore', 'remove', 'cap']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if outlier_method == 'iqr':
                df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
                df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            elif outlier_method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean = df_clean[z_scores < 3]
            elif outlier_method == 'remove':
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            elif outlier_method == 'cap':
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean.reset_index(drop=True)

def validate_dataframe(df, check_duplicates=True, check_types=True):
    """
    Validate a DataFrame for common data quality issues.

    Parameters:
    df (pd.DataFrame): The DataFrame to validate.
    check_duplicates (bool): Check for duplicate rows.
    check_types (bool): Check for consistent data types.

    Returns:
    dict: Dictionary containing validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': 0,
        'type_issues': {}
    }
    
    if check_duplicates:
        validation_results['duplicate_rows'] = df.duplicated().sum()
    
    if check_types:
        for col in df.columns:
            unique_types = df[col].apply(type).unique()
            if len(unique_types) > 1:
                validation_results['type_issues'][col] = [str(t) for t in unique_types]
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df, missing_strategy='median', outlier_method='cap')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validation = validate_dataframe(cleaned_df)
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")import csv
import sys

def clean_csv(input_file, output_file):
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            rows = list(reader)
        
        cleaned_rows = []
        for row in rows:
            cleaned_row = [cell.strip().replace('\n', ' ') for cell in row]
            cleaned_rows.append(cleaned_row)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(cleaned_rows)
        
        print(f"Cleaned data saved to {output_file}")
        return True
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_csv(input_file, output_file)