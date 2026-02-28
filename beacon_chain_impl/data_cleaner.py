import pandas as pd
import numpy as np

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

def clean_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_cols:
        df = normalize_minmax(df, col)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Original shape: {pd.read_csv(input_path).shape}")
    print(f"Cleaned shape: {df.shape}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')import numpy as np
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
        return df[column].apply(lambda x: 0.5)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    required_checks = [
        (lambda: not df.empty, "DataFrame is empty"),
        (lambda: df.isnull().sum().sum() == 0, "DataFrame contains null values"),
        (lambda: all(df.dtypes != object), "DataFrame contains non-numeric columns")
    ]
    for check, message in required_checks:
        if not check():
            raise ValueError(f"Validation failed: {message}")
    return True
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Cleans a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    cleaned_df = df.copy()

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {original_shape[0] - cleaned_df.shape[0]} duplicate rows.")

    if fill_missing is not None:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        if fill_missing == 'mean':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        elif fill_missing == 'median':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        elif fill_missing == 'zero':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)
        print(f"Filled missing values in numeric columns using method: {fill_missing}")

    print(f"Data cleaned. Original shape: {original_shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, np.nan, 5],
        'B': [10, np.nan, 10, 40, 50],
        'C': ['x', 'y', 'x', 'y', 'z']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    cleaned = clean_dataframe(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)