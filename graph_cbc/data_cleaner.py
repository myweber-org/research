import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
        subset (list): Columns to consider for duplicates (optional)
        keep (str): Which duplicates to keep: 'first', 'last', or False
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        final_rows = len(df_clean)
        
        duplicates_removed = initial_rows - final_rows
        
        if output_file:
            df_clean.to_csv(output_file, index=False)
            print(f"Cleaned data saved to: {output_file}")
        
        print(f"Initial rows: {initial_rows}")
        print(f"Final rows: {final_rows}")
        print(f"Duplicates removed: {duplicates_removed}")
        
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    cleaned_df = remove_duplicates(input_file, output_file)
    print("Data cleaning completed successfully.")import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (bool): Whether to fill missing values. Default is False.
        fill_value: Value to use for filling missing data. Default is 0.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def process_data_file(file_path, output_path=None):
    """
    Process a data file by loading, cleaning, and saving it.
    
    Args:
        file_path (str): Path to input data file.
        output_path (str): Path to save cleaned data. If None, returns DataFrame.
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")
        
        cleaned_df = clean_dataframe(df, drop_duplicates=True, fill_missing=True)
        
        is_valid, message = validate_dataframe(cleaned_df)
        if not is_valid:
            print(f"Validation warning: {message}")
        
        if output_path:
            if output_path.endswith('.csv'):
                cleaned_df.to_csv(output_path, index=False)
            elif output_path.endswith('.xlsx'):
                cleaned_df.to_excel(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return cleaned_df
            
    except Exception as e:
        print(f"Error processing file: {e}")
        return None
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Strategy to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    dict: Validation results with keys 'is_valid' and 'message'.
    """
    validation_result = {'is_valid': True, 'message': 'DataFrame is valid'}
    
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['message'] = 'DataFrame is empty'
        return validation_result
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['message'] = f'Missing required columns: {missing_columns}'
    
    return validation_result

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {validation['message']}")
import pandas as pd

def clean_dataset(df):
    """
    Remove duplicate rows and standardize column names.
    
    Args:
        df (pd.DataFrame): Input dataframe to clean.
    
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    return df_clean

def validate_dataframe(df):
    """
    Perform basic validation on dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe to validate.
    
    Returns:
        dict: Dictionary containing validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Product Name': ['A', 'B', 'A', 'C', 'B'],
        'Price': [100, 200, 100, 300, 200],
        'Quantity': [5, 3, 5, 2, 3]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataframe(df))
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'zero').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing and cleaned_df.isnull().sum().any():
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_strategy == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_strategy == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                elif fill_strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError("Invalid fill_strategy. Choose from 'mean', 'median', 'mode', or 'zero'.")
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_strategy}: {fill_value}")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
            print(f"Filled missing values in categorical column '{col}' with 'Unknown'")
    
    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate the dataset for required columns and basic integrity.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("Dataset is empty.")
        return False
    
    print("Dataset validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10.5, np.nan, 15.2, 20.1, np.nan, 10.5, 18.7],
        'category': ['A', 'B', 'A', np.nan, 'C', 'A', 'B'],
        'score': [85, 92, 78, np.nan, 88, 85, 90]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nCleaning dataset...")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean')
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    validation_result = validate_dataset(cleaned_df, required_columns=['id', 'value', 'category'])
    print(f"\nValidation result: {validation_result}")
import pandas as pd
import numpy as np

def clean_dataset(df, text_columns=None, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing text.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    text_columns (list): List of column names containing text data
    fill_strategy (str): Strategy for filling numeric missing values ('mean', 'median', 'zero')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    # Handle missing values in numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if cleaned_df[col].isnull().any():
            if fill_strategy == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_strategy == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            elif fill_strategy == 'zero':
                cleaned_df[col].fillna(0, inplace=True)
    
    # Standardize text columns
    if text_columns:
        for col in text_columns:
            if col in cleaned_df.columns:
                # Convert to string, strip whitespace, and convert to lowercase
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
                # Replace empty strings with NaN then fill with 'unknown'
                cleaned_df[col].replace(['', 'nan', 'none'], np.nan, inplace=True)
                cleaned_df[col].fillna('unknown', inplace=True)
    
    # Remove duplicate rows
    cleaned_df.drop_duplicates(inplace=True)
    
    # Reset index after cleaning
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
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
        validation_results['errors'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_cols}')
    
    # Check for all-null columns
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        validation_results['warnings'].append(f'Columns with all null values: {null_columns}')
    
    # Check data types
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            if unique_count == 1:
                validation_results['warnings'].append(f'Column "{col}" has only one unique value')
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', '', 'Eve'],
        'age': [25, 30, None, 35, 40],
        'score': [85.5, 92.0, 78.5, None, 88.0],
        'department': ['IT', 'HR', 'IT', 'Finance', 'HR']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    text_cols = ['name', 'department']
    cleaned_df = clean_dataset(df, text_columns=text_cols, fill_strategy='mean')
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age'])
    print("Validation Results:")
    print(f"Is Valid: {validation['is_valid']}")
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")import pandas as pd
import numpy as np
from scipy import stats

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    missing_strategy (str): Strategy for missing values ('mean', 'median', 'mode')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    for column in cleaned_df.select_dtypes(include=[np.number]).columns:
        if cleaned_df[column].isnull().any():
            if missing_strategy == 'mean':
                cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            elif missing_strategy == 'median':
                cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            elif missing_strategy == 'mode':
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    # Remove outliers using Z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(cleaned_df[numeric_cols]))
    
    outlier_mask = (z_scores < outlier_threshold).all(axis=1)
    cleaned_df = cleaned_df[outlier_mask].reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def get_data_summary(df):
    """
    Generate summary statistics for the dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {}
    }
    return summary

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 60],
        'C': [100, 200, 300, 400, 500, 600]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, missing_strategy='median', outlier_threshold=2.5)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")
    
    summary = get_data_summary(cleaned)
    print(f"\nData shape: {summary['shape']}")