import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for nulls. 
                                          If None, checks all columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove rows with null values
    if columns_to_check is None:
        cleaned_df = cleaned_df.dropna()
    else:
        cleaned_df = cleaned_df.dropna(subset=columns_to_check)
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        dict: Validation results with status and messages.
    """
    validation_result = {
        'is_valid': True,
        'messages': []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['messages'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['messages'].append(f'Missing columns: {missing_columns}')
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        validation_result['messages'].append(f'Null values found: {null_counts[null_counts > 0].to_dict()}')
    
    return validation_result

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
        'age': [25, 30, 35, None, 28, 28],
        'score': [85.5, 92.0, 78.5, 88.0, 95.5, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    # Validate the cleaned dataset
    validation = validate_dataset(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    print("Validation Results:")
    print(f"Is Valid: {validation['is_valid']}")
    print(f"Messages: {validation['messages']}")