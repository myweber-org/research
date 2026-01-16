
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns, method='minmax'):
    df_norm = df.copy()
    for col in columns:
        if method == 'minmax':
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            df_norm[col] = (df_norm[col] - mean_val) / std_val
    return df_norm

def clean_dataset(file_path, output_path, numeric_columns):
    df = pd.read_csv(file_path)
    df_cleaned = remove_outliers_iqr(df, numeric_columns)
    df_normalized = normalize_data(df_cleaned, numeric_columns, method='zscore')
    df_normalized.to_csv(output_path, index=False)
    return df_normalized

if __name__ == "__main__":
    numeric_cols = ['age', 'income', 'score']
    cleaned_data = clean_dataset('raw_data.csv', 'cleaned_data.csv', numeric_cols)
    print(f"Data cleaning complete. Shape: {cleaned_data.shape}")
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    df = pd.read_csv(input_file)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    # Convert date columns if present
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass
    
    # Remove outliers using IQR method for numeric columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    # Save cleaned data
    df.to_csv(output_file, index=False)
    print(f"Data cleaning completed. Cleaned data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    cleaned_df = clean_csv_data(input_path, output_path)