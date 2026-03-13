
import re

def clean_string(text):
    """
    Clean and normalize a string by:
    1. Stripping leading/trailing whitespace
    2. Replacing multiple spaces with a single space
    3. Converting to lowercase
    """
    if not isinstance(text, str):
        return text
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text.lower()
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            cleaned_df = cleaned_df[mask]
    return cleaned_df.reset_index(drop=True)

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[col] = 0
    return normalized_df

def clean_dataset(filepath, numeric_columns):
    try:
        df = pd.read_csv(filepath)
        print(f"Original shape: {df.shape}")
        
        df_cleaned = remove_outliers_iqr(df, numeric_columns)
        print(f"After outlier removal: {df_cleaned.shape}")
        
        df_normalized = normalize_minmax(df_cleaned, numeric_columns)
        
        output_path = filepath.replace('.csv', '_cleaned.csv')
        df_normalized.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return df_normalized
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'feature_c': np.random.uniform(0, 1, 200)
    })
    
    sample_data.loc[10, 'feature_a'] = 500
    sample_data.loc[20, 'feature_b'] = 1000
    
    sample_data.to_csv('sample_dataset.csv', index=False)
    
    cleaned = clean_dataset('sample_dataset.csv', ['feature_a', 'feature_b', 'feature_c'])
    
    if cleaned is not None:
        print("\nCleaned data summary:")
        print(cleaned.describe())
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Load CSV data, handle missing values, and save cleaned version.
    """
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna('Unknown')
        
        df.to_csv(output_file, index=False)
        
        print(f"Cleaned data saved to: {output_file}")
        print(f"Final data shape: {df.shape}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    clean_csv_data('raw_data.csv', 'cleaned_data.csv')