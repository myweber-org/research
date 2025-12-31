
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(input_file, output_file):
    df = load_data(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')import pandas as pd

def clean_dataset(df, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicate rows and filling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        fill_strategy (str): Strategy for filling missing values. 
                            Options: 'mean', 'median', 'mode', or 'zero'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Fill missing values based on strategy
    if fill_strategy == 'mean':
        df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
    elif fill_strategy == 'median':
        df_cleaned = df_cleaned.fillna(df_cleaned.median(numeric_only=True))
    elif fill_strategy == 'mode':
        df_cleaned = df_cleaned.fillna(df_cleaned.mode().iloc[0])
    elif fill_strategy == 'zero':
        df_cleaned = df_cleaned.fillna(0)
    else:
        raise ValueError(f"Unsupported fill strategy: {fill_strategy}")
    
    return df_cleaned

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        dict: Validation results with keys: 'is_valid', 'missing_columns', 'null_counts'
    """
    result = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {}
    }
    
    # Check required columns
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            result['is_valid'] = False
            result['missing_columns'] = missing
    
    # Count null values
    null_counts = df.isnull().sum()
    result['null_counts'] = null_counts[null_counts > 0].to_dict()
    
    return result

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': [100, 200, 200, 300, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation results:")
    print(validate_dataset(df))
    
    cleaned_df = clean_dataset(df, fill_strategy='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)