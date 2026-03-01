import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: bool, whether to drop duplicate rows
        fill_method: str or value, method to fill missing values
                    ('mean', 'median', 'mode', or specific value)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method is not None:
        if fill_method == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_method == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_method == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        else:
            cleaned_df = cleaned_df.fillna(fill_method)
    else:
        cleaned_df = cleaned_df.dropna()
    
    # Remove duplicates
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
        min_rows: minimum number of rows required
    
    Returns:
        tuple: (is_valid, message)
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

def get_dataset_stats(df):
    """
    Get basic statistics about the dataset.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        dict with dataset statistics
    """
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': list(df.select_dtypes(include=['number']).columns),
        'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    return stats

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, None, 4, 5, 5],
        'B': ['x', 'y', 'z', None, 'x', 'x'],
        'C': [10.5, 20.3, 30.1, 40.7, None, 10.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nDataset statistics:")
    print(get_dataset_stats(df))
    
    # Clean the dataset
    cleaned = clean_dataset(df, drop_duplicates=True, fill_method='mean')
    print("\nCleaned dataset:")
    print(cleaned)
    
    # Validate the cleaned dataset
    is_valid, message = validate_dataset(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")