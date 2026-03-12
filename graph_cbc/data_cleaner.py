import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        remove_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str or dict): Method to fill missing values:
            - 'mean': Fill with column mean
            - 'median': Fill with column median
            - 'mode': Fill with column mode
            - dict: Column-specific fill values
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing is not None:
        if cleaned_df.isnull().sum().sum() > 0:
            if isinstance(fill_missing, dict):
                cleaned_df = cleaned_df.fillna(fill_missing)
            elif fill_missing == 'mean':
                cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
            elif fill_missing == 'median':
                cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
            elif fill_missing == 'mode':
                cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
            print(f"Filled missing values using method: {fill_missing}")
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if len(df) < min_rows:
        return False, f"Dataset must have at least {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Dataset validation passed"

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, None, 15.2, 20.1, None],
        'category': ['A', 'B', 'B', 'A', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nCleaned dataset:")
    cleaned = clean_dataset(df, remove_duplicates=True, fill_missing='mean')
    print(cleaned)
    
    is_valid, message = validate_dataset(cleaned, required_columns=['id', 'value'])
    print(f"\nValidation: {message}")