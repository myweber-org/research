
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

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for column in numeric_columns:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
            df = normalize_minmax(df, column)
    
    df = df.dropna()
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv', ['age', 'income', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaning complete. Remaining records: {len(cleaned_data)}")
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and standardizing column names.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output CSV. 
                                     If None, adds '_cleaned' suffix to input filename.
        missing_strategy (str): Strategy for handling missing values.
                               Options: 'mean', 'median', 'drop', 'zero'
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    
    # Validate input file exists
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read CSV file
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif missing_strategy == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    elif missing_strategy == 'zero':
        df = df.fillna(0)
    elif missing_strategy == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_cleaned.csv"
    else:
        output_path = Path(output_path)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    
    print(f"Data cleaning completed. Cleaned data saved to: {output_path}")
    print(f"Original shape: {df.shape}, Missing values handled: {missing_strategy}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {}
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing required columns: {missing_cols}')
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        validation_results['issues'].append(f'Found {duplicate_count} duplicate rows')
    
    # Generate summary statistics
    validation_results['summary'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': duplicate_count,
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'ID': [1, 2, 3, 4, 5],
        'Value': [10.5, None, 15.2, None, 20.1],
        'Category': ['A', 'B', 'A', 'C', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample data created for demonstration")
    print(df)
    
    # Validate the sample data
    validation = validate_dataframe(df, required_columns=['ID', 'Value'])
    print("\nValidation results:")
    print(validation)