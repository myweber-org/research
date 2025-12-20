
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list, optional): List of column names to check for duplicates.
                                       If None, checks all columns.
    fill_missing (str or dict): Method to fill missing values.
                                Can be 'mean', 'median', 'mode', or a dictionary
                                of column:value pairs for custom filling.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns.tolist()
    
    cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check, keep='first')
    
    # Handle missing values
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().any():
            if isinstance(fill_missing, dict):
                # Use custom value if provided in dictionary
                if column in fill_missing:
                    cleaned_df[column] = cleaned_df[column].fillna(fill_missing[column])
                else:
                    # If column not in dictionary, use column mean for numeric, mode for categorical
                    if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
                    else:
                        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
            elif fill_missing == 'mean':
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
                else:
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
            elif fill_missing == 'median':
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
                else:
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
            elif fill_missing == 'mode':
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
            else:
                # Default to mean for numeric, mode for categorical
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
                else:
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Dataset is valid"
import pandas as pd
import numpy as np
from datetime import datetime

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by handling missing values,
    converting data types, and removing duplicates.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Fill missing numeric values with column median
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    # Fill missing categorical values with mode
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
    
    # Convert date columns to datetime
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col])
            except (ValueError, TypeError):
                pass
    
    # Remove outliers using IQR method for numeric columns
    for col in numeric_cols:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    """
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def process_csv_file(input_path, output_path, required_columns=None):
    """
    Main function to process CSV file through cleaning pipeline.
    """
    try:
        # Read CSV file
        df = pd.read_csv(input_path)
        
        # Validate data
        validate_dataframe(df, required_columns)
        
        # Clean data
        cleaned_df = clean_dataframe(df)
        
        # Save cleaned data
        cleaned_df.to_csv(output_path, index=False)
        
        print(f"Data cleaning completed. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
        print(f"Cleaned data saved to: {output_path}")
        
        return cleaned_df
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    # Define required columns if needed
    required_cols = ['id', 'name', 'value']
    
    try:
        result = process_csv_file(input_file, output_file, required_cols)
        print("Processing completed successfully")
    except Exception as e:
        print(f"Processing failed: {e}")