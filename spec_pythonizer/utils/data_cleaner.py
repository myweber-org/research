
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        remove_duplicates (bool): Whether to remove duplicate rows.
        fill_missing (str or dict): Strategy to fill missing values.
            Options: 'mean', 'median', 'mode', or a dictionary of column:value.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df.fillna(fill_missing, inplace=True)
        elif fill_missing == 'mean':
            cleaned_df.fillna(cleaned_df.mean(numeric_only=True), inplace=True)
        elif fill_missing == 'median':
            cleaned_df.fillna(cleaned_df.median(numeric_only=True), inplace=True)
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else '', inplace=True)
    
    # Remove duplicates
    if remove_duplicates:
        cleaned_df.drop_duplicates(inplace=True)
    
    # Reset index after cleaning
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of columns that must be present.
    
    Returns:
        dict: Dictionary with validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_count': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing_columns
        validation_results['all_required_present'] = len(missing_columns) == 0
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, None, 4, 2],
        'B': [5, None, 7, 8, 5],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation results:")
    print(validate_dataset(df))
    
    cleaned = clean_dataset(df, remove_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nCleaned validation results:")
    print(validate_dataset(cleaned))import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, overwrites input file
        subset (list, optional): Columns to consider for identifying duplicates
        keep (str): Which duplicates to keep - 'first', 'last', or False (remove all)
    
    Returns:
        int: Number of duplicate rows removed
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        final_rows = len(df_clean)
        
        duplicates_removed = initial_rows - final_rows
        
        if output_file is None:
            output_file = input_file
        
        df_clean.to_csv(output_file, index=False)
        
        print(f"Removed {duplicates_removed} duplicate rows")
        print(f"Original rows: {initial_rows}, Cleaned rows: {final_rows}")
        print(f"Saved to: {output_file}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return -1
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty")
        return -1
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return -1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = remove_duplicates(input_file, output_file)
    
    if result >= 0:
        sys.exit(0)
    else:
        sys.exit(1)