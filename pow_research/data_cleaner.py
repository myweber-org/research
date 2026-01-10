
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict): Optional dictionary mapping old column names to new ones
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    duplicates_removed = initial_rows - len(cleaned_df)
    
    # Standardize column names
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Convert column names to lowercase with underscores
    cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(' ', '_')
    
    # Remove leading/trailing whitespace from string columns
    string_columns = cleaned_df.select_dtypes(include=['object']).columns
    for col in string_columns:
        cleaned_df[col] = cleaned_df[col].str.strip()
    
    # Replace empty strings with NaN
    cleaned_df = cleaned_df.replace(r'^\s*$', np.nan, regex=True)
    
    # Print cleaning summary
    print(f"Cleaning complete:")
    print(f"  - Removed {duplicates_removed} duplicate rows")
    print(f"  - Final dataset shape: {cleaned_df.shape}")
    print(f"  - Columns: {list(cleaned_df.columns)}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of columns that must be present
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
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
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for null values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        validation_results['warnings'].append(f"Found {total_nulls} null values in dataset")
        validation_results['summary']['null_counts'] = null_counts[null_counts > 0].to_dict()
    
    # Check data types
    dtype_summary = df.dtypes.to_dict()
    validation_results['summary']['dtypes'] = dtype_summary
    
    # Calculate basic statistics
    validation_results['summary']['shape'] = df.shape
    validation_results['summary']['memory_usage'] = df.memory_usage(deep=True).sum()
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Customer ID': [1, 2, 2, 3, 4],
        'First Name': ['John', 'Jane', 'Jane', 'Bob', 'Alice'],
        'Last Name': ['Doe', 'Smith', 'Smith', 'Johnson', 'Brown'],
        'Email': ['john@example.com', 'jane@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com'],
        'Age': [25, 30, 30, 35, 28],
        'City': ['New York', 'Los Angeles', 'Los Angeles', 'Chicago', 'Boston']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    print("\n" + "="*50 + "\n")
    validation = validate_dataframe(cleaned_df, required_columns=['customer_id', 'email'])
    print("Validation Results:")
    print(f"Is valid: {validation['is_valid']}")
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")
    print(f"Summary: {validation['summary']}")
import pandas as pd
import numpy as np

def remove_outliers(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return df[z_scores < threshold]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv')
    cleaned_data.to_csv('cleaned_data.csv', index=False)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. Can be 'mean', 
                                   'median', 'mode', or a dictionary of column:value pairs.
                                   If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, unique_columns=None):
    """
    Validate a DataFrame for required columns and unique constraints.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        unique_columns (list): List of column names that should have unique values.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if unique_columns:
        for col in unique_columns:
            if col in df.columns and df[col].duplicated().any():
                return False, f"Column '{col}' contains duplicate values"
    
    return True, "Dataset is valid"