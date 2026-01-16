import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: Input pandas DataFrame
        drop_duplicates: Whether to remove duplicate rows (default: True)
        fill_missing: Whether to fill missing values (default: False)
        fill_value: Value to use for filling missing data (default: 0)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_rows = initial_rows - len(cleaned_df)
        print(f"Removed {removed_rows} duplicate rows")
    
    if fill_missing:
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        missing_after = cleaned_df.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values")
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a dataset for required columns and data integrity.
    
    Args:
        df: Input pandas DataFrame
        required_columns: List of required column names (default: None)
    
    Returns:
        Boolean indicating if dataset is valid
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("Dataset is empty")
        return False
    
    return True

def get_dataset_summary(df):
    """
    Generate a summary of the dataset.
    
    Args:
        df: Input pandas DataFrame
    
    Returns:
        Dictionary containing dataset summary
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve'],
        'score': [85, 90, 90, 88, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nDataset summary:")
    print(get_dataset_summary(df))
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0)
    print("\nCleaned dataset:")
    print(cleaned)
    
    is_valid = validate_dataset(cleaned, required_columns=['id', 'name', 'score'])
    print(f"\nDataset is valid: {is_valid}")