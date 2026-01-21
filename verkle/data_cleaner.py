
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
        original_shape = df.shape
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df = remove_outliers_iqr(df, col)
        
        df = df.dropna()
        final_shape = df.shape
        
        df.to_csv(output_path, index=False)
        
        print(f"Original dataset shape: {original_shape}")
        print(f"Cleaned dataset shape: {final_shape}")
        print(f"Removed {original_shape[0] - final_shape[0]} rows")
        print(f"Cleaned data saved to: {output_path}")
        
        return df
    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    cleaned_df = clean_dataset(input_file, output_file)import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: Optional[str] = None,
    missing_strategy: str = "mean",
    convert_types: bool = True
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and converting data types.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV (optional)
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'drop', 'zero')
        convert_types: Whether to automatically convert data types
    
    Returns:
        Cleaned DataFrame
    """
    
    df = pd.read_csv(input_path)
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif missing_strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif missing_strategy == "zero":
        df[numeric_cols] = df[numeric_cols].fillna(0)
    elif missing_strategy == "drop":
        df = df.dropna(subset=numeric_cols)
    
    if convert_types:
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except (ValueError, TypeError):
                        pass
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    final_shape = df.shape
    print(f"Final data shape: {final_shape}")
    print(f"Rows removed: {original_shape[0] - final_shape[0]}")
    
    return df

def validate_dataframe(df: pd.DataFrame, required_cols: list = None) -> bool:
    """
    Validate DataFrame for common data quality issues.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
    
    Returns:
        Boolean indicating if validation passed
    """
    
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if required_cols:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing columns: {missing_cols}")
            return False
    
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        print(f"Warning: Found {duplicate_rows} duplicate rows")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            print(f"Warning: Column '{col}' has {df[col].isnull().sum()} missing values")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, None, 15.2, None, 20.1],
        'category': ['A', 'B', None, 'A', 'C'],
        'date': ['2023-01-01', '2023-01-02', None, '2023-01-04', '2023-01-05']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data(
        'test_data.csv',
        'cleaned_data.csv',
        missing_strategy='mean',
        convert_types=True
    )
    
    is_valid = validate_dataframe(cleaned_df, required_cols=['id', 'value'])
    print(f"Data validation passed: {is_valid}")