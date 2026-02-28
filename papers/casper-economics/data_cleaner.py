
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def standardize_column(df, column):
    """
    Standardize a column to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to standardize.
    
    Returns:
    pd.DataFrame: DataFrame with standardized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = df[column].mean()
    std_val = df[column].std()
    
    if std_val == 0:
        return df
    
    df_copy = df.copy()
    df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    strategy (str): Imputation strategy ('mean', 'median', or 'drop').
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df.dropna(subset=numeric_cols)
    elif strategy == 'mean':
        impute_values = df[numeric_cols].mean()
    elif strategy == 'median':
        impute_values = df[numeric_cols].median()
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    df_filled = df.copy()
    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(impute_values)
    
    return df_filled

def clean_dataset(df, numeric_columns=None, outlier_removal=True, standardization=True, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_columns (list): List of numeric columns to process. If None, all numeric columns are used.
    outlier_removal (bool): Whether to remove outliers.
    standardization (bool): Whether to standardize columns.
    missing_strategy (str): Strategy for handling missing values.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if outlier_removal:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    if standardization:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = standardize_column(cleaned_df, col)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [5, np.nan, 15, 20, 25, 30, 35, 40, 45, 50]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataset(df, outlier_removal=True, standardization=True)
    print("Cleaned DataFrame:")
    print(cleaned)
import pandas as pd
import numpy as np

def clean_data(input_file, output_file):
    df = pd.read_csv(input_file)
    
    df = df.dropna()
    
    df = df.drop_duplicates()
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    print(f"Original rows: {len(pd.read_csv(input_file))}, Cleaned rows: {len(df)}")

if __name__ == "__main__":
    clean_data("raw_data.csv", "cleaned_data.csv")import csv
import sys

def clean_csv(input_file, output_file, key_column):
    """
    Remove duplicate rows based on a key column and convert numeric columns.
    """
    seen = set()
    cleaned_rows = []
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            
            for row in reader:
                key = row.get(key_column)
                if key is None:
                    continue
                
                if key not in seen:
                    seen.add(key)
                    processed_row = {}
                    for field in fieldnames:
                        value = row[field]
                        if value.replace('.', '', 1).isdigit():
                            if '.' in value:
                                processed_row[field] = float(value)
                            else:
                                processed_row[field] = int(value)
                        else:
                            processed_row[field] = value
                    cleaned_rows.append(processed_row)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned_rows)
            
        print(f"Cleaned data saved to {output_file}")
        print(f"Removed {len(seen) - len(cleaned_rows)} duplicate rows")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python data_cleaner.py <input.csv> <output.csv> <key_column>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key_column = sys.argv[3]
    
    clean_csv(input_file, output_file, key_column)