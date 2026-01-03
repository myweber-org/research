
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing='mean', remove_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_check (list): List of columns to check for missing values, defaults to all columns
    fill_missing (str): Method to fill missing values - 'mean', 'median', 'mode', or 'drop'
    remove_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    # Remove duplicates if requested
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    # Handle missing values
    if columns_to_check is None:
        columns_to_check = df_clean.columns
    
    missing_counts = df_clean[columns_to_check].isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0].index.tolist()
    
    if columns_with_missing:
        print(f"Columns with missing values: {columns_with_missing}")
        
        for column in columns_with_missing:
            if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df_clean[column]):
                fill_value = df_clean[column].mean()
                df_clean[column] = df_clean[column].fillna(fill_value)
                print(f"Filled missing values in '{column}' with mean: {fill_value:.2f}")
            
            elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df_clean[column]):
                fill_value = df_clean[column].median()
                df_clean[column] = df_clean[column].fillna(fill_value)
                print(f"Filled missing values in '{column}' with median: {fill_value:.2f}")
            
            elif fill_missing == 'mode':
                fill_value = df_clean[column].mode()[0] if not df_clean[column].mode().empty else None
                df_clean[column] = df_clean[column].fillna(fill_value)
                print(f"Filled missing values in '{column}' with mode: {fill_value}")
            
            elif fill_missing == 'drop':
                df_clean = df_clean.dropna(subset=[column])
                print(f"Dropped rows with missing values in '{column}'")
            
            else:
                print(f"Warning: Could not fill missing values in '{column}' with method '{fill_missing}'")
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {df_clean.shape}")
    return df_clean

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    numeric_columns (list): List of columns that should be numeric
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'non_numeric_columns': [],
        'empty': df.empty,
        'shape': df.shape
    }
    
    # Check required columns
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    # Check numeric columns
    if numeric_columns:
        non_numeric = []
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric.append(col)
        
        if non_numeric:
            validation_results['non_numeric_columns'] = non_numeric
            validation_results['is_valid'] = False
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data with some issues
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
        'age': [25, 30, None, 35, 40, 40, 45],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, fill_missing='mean', remove_duplicates=True)
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    validation = validate_dataframe(
        cleaned_df,
        required_columns=['id', 'name', 'age', 'score'],
        numeric_columns=['age', 'score']
    )
    
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")