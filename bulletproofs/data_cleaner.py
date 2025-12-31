import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output CSV. 
                                   If None, adds '_cleaned' suffix to input filename.
        strategy (str): Strategy for handling missing values: 
                       'mean', 'median', 'mode', or 'drop'
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    
    # Validate input file exists
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read CSV file
    df = pd.read_csv(input_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    # Remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values based on strategy
    if strategy == 'drop':
        df = df.dropna()
    elif strategy in ['mean', 'median']:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if strategy == 'mean':
                fill_value = df[col].mean()
            else:  # median
                fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
    elif strategy == 'mode':
        for col in df.columns:
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value[0])
    
    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_cleaned.csv"
    else:
        output_path = Path(output_path)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Cleaned data saved to: {output_path}")
    
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
        'errors': [],
        'warnings': []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(
                f"Missing required columns: {missing_columns}"
            )
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            validation_results['warnings'].append(
                f"Column '{col}' contains infinite values"
            )
    
    # Check data types consistency
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for mixed types in object columns
            unique_types = set(type(val) for val in df[col].dropna())
            if len(unique_types) > 1:
                validation_results['warnings'].append(
                    f"Column '{col}' has mixed data types: {unique_types}"
                )
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, np.nan, 15.2, 20.1, np.nan, 20.1],
        'category': ['A', 'B', 'A', 'C', 'B', 'C'],
        'score': [100, 200, np.nan, 400, 500, 400]
    }
    
    # Create sample DataFrame
    df_sample = pd.DataFrame(sample_data)
    
    # Save sample data to CSV
    sample_path = Path("sample_data.csv")
    df_sample.to_csv(sample_path, index=False)
    
    # Clean the data
    try:
        cleaned_df = clean_csv_data(
            input_path="sample_data.csv",
            strategy='mean'
        )
        
        # Validate cleaned data
        validation = validate_dataframe(
            cleaned_df,
            required_columns=['id', 'value', 'category', 'score']
        )
        
        print("\nValidation Results:")
        print(f"Is valid: {validation['is_valid']}")
        if validation['errors']:
            print("Errors:", validation['errors'])
        if validation['warnings']:
            print("Warnings:", validation['warnings'])
        
        # Clean up sample file
        sample_path.unlink(missing_ok=True)
        
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        # Clean up sample file on error
        sample_path.unlink(missing_ok=True)