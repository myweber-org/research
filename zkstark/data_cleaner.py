
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values with column means.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                mean_value = cleaned_df[column].mean()
                cleaned_df[column].fillna(mean_value, inplace=True)
                print(f"Filled missing values in column '{column}' with mean: {mean_value:.2f}")
    
    return cleaned_df

def validate_dataframe(df, check_missing=True, check_types=True):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    check_missing (bool): Check for missing values.
    check_types (bool): Check column data types.
    
    Returns:
    dict: Validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'column_types': {}
    }
    
    if check_missing:
        missing_counts = df.isnull().sum()
        validation_results['missing_values'] = missing_counts[missing_counts > 0].to_dict()
    
    if check_types:
        validation_results['column_types'] = df.dtypes.astype(str).to_dict()
    
    return validation_results

def process_csv_file(input_path, output_path=None):
    """
    Process a CSV file through the data cleaning pipeline.
    
    Parameters:
    input_path (str): Path to input CSV file.
    output_path (str): Path to save cleaned CSV file.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded data from {input_path}")
        print(f"Original shape: {df.shape}")
        
        validation = validate_dataframe(df)
        print(f"Validation results: {validation}")
        
        cleaned_df = clean_dataframe(df)
        print(f"Cleaned shape: {cleaned_df.shape}")
        
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
            print(f"Saved cleaned data to {output_path}")
        
        return cleaned_df
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10.5, 20.3, np.nan, 30.1, 40.0, np.nan],
        'category': ['A', 'B', 'A', 'A', 'C', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample DataFrame:")
    print(df)
    print("\nCleaning data...")
    
    cleaned = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned)
    print("\nFinal validation:")
    print(validation)