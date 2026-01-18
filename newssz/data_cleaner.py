import pandas as pd

def remove_duplicates(input_file, output_file, key_columns):
    """
    Load a CSV file, remove duplicate rows based on specified columns,
    and save the cleaned data to a new file.
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        df_cleaned = df.drop_duplicates(subset=key_columns, keep='first')
        final_count = len(df_cleaned)
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Data cleaning completed.")
        print(f"Initial records: {initial_count}")
        print(f"Final records: {final_count}")
        print(f"Duplicates removed: {initial_count - final_count}")
        print(f"Cleaned data saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    unique_keys = ["id", "email"]
    
    cleaned_data = remove_duplicates(input_csv, output_csv, unique_keys)
import numpy as np
import pandas as pd

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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.0)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    if std_val == 0:
        return df[column].apply(lambda x: 0.0)
    return df[column].apply(lambda x: (x - mean_val) / std_val)

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='standard'):
    cleaned_df = df.copy()
    
    if outlier_removal:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if normalization == 'minmax':
                cleaned_df[col] = normalize_minmax(cleaned_df, col)
            elif normalization == 'standard':
                cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_cleaning(df, original_df, numeric_columns):
    report = {}
    for col in numeric_columns:
        if col in df.columns and col in original_df.columns:
            report[col] = {
                'original_mean': original_df[col].mean(),
                'cleaned_mean': df[col].mean(),
                'original_std': original_df[col].std(),
                'cleaned_std': df[col].std(),
                'original_range': (original_df[col].min(), original_df[col].max()),
                'cleaned_range': (df[col].min(), df[col].max()),
                'rows_removed': len(original_df) - len(df)
            }
    return pd.DataFrame.from_dict(report, orient='index')