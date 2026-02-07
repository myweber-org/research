import pandas as pd
import numpy as np
import os

def clean_csv_data(input_file, output_file=None):
    """
    Clean a CSV file by handling missing values, duplicates,
    and standardizing column names.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Remove duplicate rows
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(df)
        
        # Handle missing values
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown', inplace=True)
        
        # Remove rows where all values are NaN (shouldn't happen after filling, but just in case)
        df.dropna(how='all', inplace=True)
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_cleaned.csv"
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        
        # Print cleaning summary
        print(f"Data cleaning completed:")
        print(f"  - Input file: {input_file}")
        print(f"  - Output file: {output_file}")
        print(f"  - Rows processed: {initial_count}")
        print(f"  - Duplicates removed: {duplicates_removed}")
        print(f"  - Missing values filled: {df.isnull().sum().sum()} remaining")
        print(f"  - Final row count: {len(df)}")
        
        return df, output_file
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None, None

def validate_dataframe(df):
    """
    Perform basic validation on a DataFrame.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None.")
        return False
    
    validation_results = {
        'has_data': not df.empty,
        'row_count': len(df),
        'column_count': len(df.columns),
        'has_duplicates': df.duplicated().any(),
        'missing_values': df.isnull().sum().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    print("Data validation results:")
    for key, value in validation_results.items():
        print(f"  - {key}: {value}")
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
        'Age': [25, 30, None, 35, 40],
        'City': ['NYC', 'LA', 'NYC', 'LA', None],
        'Score': [85.5, 92.0, 78.5, 85.5, 92.0]
    }
    
    # Create a temporary CSV for testing
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    # Clean the data
    cleaned_df, output_path = clean_csv_data('test_data.csv')
    
    if cleaned_df is not None:
        # Validate the cleaned data
        validate_dataframe(cleaned_df)
    
    # Clean up test file
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')
    if os.path.exists('test_data_cleaned.csv'):
        os.remove('test_data_cleaned.csv')import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (bool): Whether to fill missing values
    fill_value: Value to use for filling missing data
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    method (str): Method for outlier detection ('iqr' or 'zscore')
    threshold (float): Threshold for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data = df[column].dropna()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        z_scores = (df[column] - mean) / std
        mask = abs(z_scores) <= threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]