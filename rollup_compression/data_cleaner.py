import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Load a CSV file, clean missing values, and save the cleaned data.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the cleaned CSV file.
        missing_strategy (str): Strategy for handling missing values.
            Options: 'mean', 'median', 'drop', 'zero'.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded data from {input_file}. Shape: {df.shape}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values in the dataset.")
            
            if missing_strategy == 'mean':
                # Fill numeric columns with mean
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                print("Filled missing values with column means.")
                
            elif missing_strategy == 'median':
                # Fill numeric columns with median
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                print("Filled missing values with column medians.")
                
            elif missing_strategy == 'drop':
                # Drop rows with any missing values
                df = df.dropna()
                print(f"Dropped rows with missing values. New shape: {df.shape}")
                
            elif missing_strategy == 'zero':
                # Fill all missing values with 0
                df = df.fillna(0)
                print("Filled missing values with zeros.")
                
            else:
                print(f"Unknown strategy: {missing_strategy}. Using 'mean' as default.")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        else:
            print("No missing values found in the dataset.")
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}. Shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df, required_columns=None):
    """
    Validate the cleaned dataframe for basic data quality.
    
    Args:
        df (pd.DataFrame): Dataframe to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        dict: Dictionary with validation results.
    """
    validation_results = {
        'is_valid': True,
        'issues': []
    }
    
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty or None')
        return validation_results
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing required columns: {missing_cols}')
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        validation_results['issues'].append(f'Found {inf_count} infinite values')
    
    # Check data types consistency
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for mixed types in object columns
            unique_types = set([type(x).__name__ for x in df[col].dropna()])
            if len(unique_types) > 1:
                validation_results['issues'].append(f'Column {col} has mixed types: {unique_types}')
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    # Clean the data using mean imputation
    cleaned_df = clean_csv_data(input_csv, output_csv, missing_strategy='mean')
    
    if cleaned_df is not None:
        # Validate the cleaned data
        validation = validate_data(cleaned_df)
        
        if validation['is_valid']:
            print("Data validation passed.")
        else:
            print("Data validation failed. Issues:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        # Show basic statistics
        print("\nBasic statistics:")
        print(cleaned_df.describe())
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (bool): Whether to fill missing values
    strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'zero')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_rows = initial_rows - len(cleaned_df)
        print(f"Removed {removed_rows} duplicate rows")
    
    if fill_missing and cleaned_df.isnull().sum().any():
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                if strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 0
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    fill_value = cleaned_df[column].mean()
                
                missing_count = cleaned_df[column].isnull().sum()
                if missing_count > 0:
                    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                    print(f"Filled {missing_count} missing values in column '{column}' with {strategy} value: {fill_value}")
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if len(df) < min_rows:
        print(f"Dataset has only {len(df)} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def main():
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10.5, 20.3, np.nan, 30.1, 40.7, np.nan],
        'category': ['A', 'B', 'A', 'A', 'C', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nDataset info:")
    print(df.info())
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean')
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    is_valid = validate_dataset(cleaned_df, required_columns=['id', 'value', 'category'], min_rows=3)
    print(f"\nDataset validation: {'PASS' if is_valid else 'FAIL'}")

if __name__ == "__main__":
    main()