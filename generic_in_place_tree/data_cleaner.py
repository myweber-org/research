
import pandas as pd

def clean_dataset(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values. Default True.
    column_case (str): Target case for column names ('lower', 'upper', 'title'). Default 'lower'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    else:
        # Fill numeric columns with median, object columns with mode
        for col in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
    
    # Standardize column names
    if column_case == 'lower':
        cleaned_df.columns = cleaned_df.columns.str.lower()
    elif column_case == 'upper':
        cleaned_df.columns = cleaned_df.columns.str.upper()
    elif column_case == 'title':
        cleaned_df.columns = cleaned_df.columns.str.title()
    
    # Remove leading/trailing whitespace from column names
    cleaned_df.columns = cleaned_df.columns.str.strip()
    
    # Replace spaces with underscores in column names
    cleaned_df.columns = cleaned_df.columns.str.replace(' ', '_')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        validation_result['is_valid'] = False
        validation_result['errors'].append('Input is not a pandas DataFrame')
        return validation_result
    
    # Check for empty DataFrame
    if df.empty:
        validation_result['warnings'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Check for duplicate columns
    if len(df.columns) != len(set(df.columns)):
        validation_result['warnings'].append('Duplicate column names detected')
    
    return validation_result

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, None, 30, 35],
        'City': ['New York', 'London', 'Paris', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, drop_na=False, column_case='lower')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned, required_columns=['name', 'age'])
    print("Validation Results:")
    print(f"Is valid: {validation['is_valid']}")
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")