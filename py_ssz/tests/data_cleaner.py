import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str or dict): Method to fill missing values. 
                                Can be 'mean', 'median', 'mode', or a dictionary of column:value pairs.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            for column, value in fill_missing.items():
                if column in cleaned_df.columns:
                    cleaned_df[column] = cleaned_df[column].fillna(value)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for column in cleaned_df.columns:
                if cleaned_df[column].dtype == 'object':
                    mode_value = cleaned_df[column].mode()
                    if not mode_value.empty:
                        cleaned_df[column] = cleaned_df[column].fillna(mode_value.iloc[0])
    
    return cleaned_df

def validate_data(df, required_columns=None, column_types=None):
    """
    Validate DataFrame structure and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    column_types (dict): Dictionary mapping column names to expected data types.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if column_types is not None:
        for column, expected_type in column_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if not actual_type.startswith(expected_type):
                    return False, f"Column '{column}' has type {actual_type}, expected {expected_type}"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve'],
        'age': [25, 30, 30, None, 35],
        'score': [85.5, 92.0, 92.0, 78.5, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    # Validate the cleaned data
    required_cols = ['id', 'name', 'age', 'score']
    col_types = {'id': 'int', 'name': 'object', 'age': 'int', 'score': 'float'}
    
    is_valid, message = validate_data(cleaned, required_columns=required_cols, column_types=col_types)
    print(f"Validation result: {is_valid}")
    print(f"Message: {message}")