import pandas as pd
import argparse
import sys

def remove_duplicates(input_file, output_file, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        subset (list, optional): Columns to consider for duplicates
        keep (str): Which duplicate to keep - 'first', 'last', or False
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        final_count = len(df_cleaned)
        
        df_cleaned.to_csv(output_file, index=False)
        
        duplicates_removed = initial_count - final_count
        print(f"Successfully removed {duplicates_removed} duplicate rows")
        print(f"Original rows: {initial_count}, Cleaned rows: {final_count}")
        print(f"Cleaned data saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Remove duplicate rows from CSV files')
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('output', help='Output CSV file path')
    parser.add_argument('--columns', nargs='+', help='Columns to check for duplicates')
    parser.add_argument('--keep', choices=['first', 'last', 'none'], 
                       default='first', help='Which duplicate to keep')
    
    args = parser.parse_args()
    
    keep_value = 'first' if args.keep == 'first' else 'last' if args.keep == 'last' else False
    
    remove_duplicates(args.input, args.output, args.columns, keep_value)

if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = normalize_minmax(cleaned_df, col)
            cleaned_df = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def handle_missing_values(df, strategy='mean'):
    df_clean = df.copy()
    
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if df_clean[col].isnull().any():
            if strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature3': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, ['feature1', 'feature2', 'feature3'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)