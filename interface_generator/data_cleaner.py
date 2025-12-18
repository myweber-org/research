
import pandas as pd
import numpy as np
from datetime import datetime

def clean_dataset(input_file, output_file):
    """
    Load a dataset, remove duplicate rows, standardize date formats,
    and fill missing numeric values with column medians.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(df)
        
        # Standardize date columns (assuming columns with 'date' in name)
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        
        # Print summary statistics
        print(f"Data cleaning completed:")
        print(f"  - Removed {duplicates_removed} duplicate rows")
        print(f"  - Processed {len(date_columns)} date columns")
        print(f"  - Filled missing values in {len(numeric_cols)} numeric columns")
        print(f"  - Cleaned data saved to: {output_file}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Perform basic validation on a dataframe.
    """
    if df is None or df.empty:
        return False
    
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'text_columns': len(df.select_dtypes(include=['object']).columns)
    }
    
    print("Data validation results:")
    for key, value in validation_results.items():
        print(f"  - {key}: {value}")
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    
    cleaned_df = clean_dataset(input_path, output_path)
    
    if cleaned_df is not None:
        validate_dataframe(cleaned_df)import numpy as np
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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col + '_normalized'] = normalize_minmax(cleaned_df, col)
            cleaned_df[col + '_standardized'] = standardize_zscore(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'feature_a': [10, 12, 12, 13, 12, 50, 11, 12, 9, 10],
        'feature_b': [100, 120, 130, 110, 115, 500, 105, 125, 95, 100]
    }
    df = pd.DataFrame(sample_data)
    result = clean_dataset(df, ['feature_a', 'feature_b'])
    print("Original shape:", df.shape)
    print("Cleaned shape:", result.shape)
    print(result.head())