
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, missing_strategy='drop'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path for cleaned CSV file (optional)
        missing_strategy: Strategy for handling missing values ('drop', 'mean', 'median', 'mode')
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    original_rows = len(df)
    original_columns = len(df.columns)
    
    df = df.drop_duplicates()
    
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif missing_strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif missing_strategy == 'mode':
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    cleaned_rows = len(df)
    duplicates_removed = original_rows - cleaned_rows
    
    print(f"Data cleaning completed:")
    print(f"  Original rows: {original_rows}")
    print(f"  Cleaned rows: {cleaned_rows}")
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Columns: {original_columns}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df: pandas.DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        dict: Validation results
    """
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('DataFrame is empty')
        return validation_results
    
    validation_results['stats']['row_count'] = len(df)
    validation_results['stats']['column_count'] = len(df.columns)
    validation_results['stats']['missing_values'] = df.isnull().sum().sum()
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        validation_results['warnings'].append('No numeric columns found')
    
    for col in numeric_cols:
        if df[col].std() == 0:
            validation_results['warnings'].append(f'Column {col} has zero variance')
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve'],
        'age': [25, 30, None, 40, 35, 35],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv', 'mean')
    
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    print("\nValidation results:", validation)