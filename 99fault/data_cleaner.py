
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    """Normalize column using min-max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(df, numeric_columns):
    """Main data cleaning pipeline."""
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_column(cleaned_df, col)
    
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

def save_cleaned_data(df, output_path):
    """Save cleaned data to CSV."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ["age", "income", "score"]
    
    raw_data = load_dataset(input_file)
    cleaned_data = clean_data(raw_data, numeric_cols)
    save_cleaned_data(cleaned_data, output_file)
    
    print(f"Original data shape: {raw_data.shape}")
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Cleaned data saved to: {output_file}")
import pandas as pd
import numpy as np

def clean_dataset(df, deduplicate=True, convert_types=True):
    """
    Clean a pandas DataFrame by removing duplicates and converting data types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    deduplicate (bool): If True, remove duplicate rows.
    convert_types (bool): If True, convert columns to optimal data types.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if convert_types:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                print(f"Converted column '{col}' to datetime.")
            except (ValueError, TypeError):
                try:
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='ignore')
                    if cleaned_df[col].dtype != 'object':
                        print(f"Converted column '{col}' to numeric.")
                except (ValueError, TypeError):
                    pass
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    print(f"Cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {},
        'data_types': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['is_valid'] = False
            validation_results['missing_columns'] = missing
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_results['null_counts'][col] = null_count
        
        validation_results['data_types'][col] = str(df[col].dtype)
    
    return validation_results

def sample_usage():
    """Demonstrate usage of the data cleaning functions."""
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David'],
        'score': ['85', '92', '92', '78', '95'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-04']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, deduplicate=True, convert_types=True)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_columns=['id', 'name', 'score'])
    print("\nValidation Results:")
    print(validation)

if __name__ == "__main__":
    sample_usage()