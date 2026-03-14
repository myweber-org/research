import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. 
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing is not None:
        if fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        elif isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        else:
            print(f"Warning: Unsupported fill_missing method '{fill_missing}'. Skipping fill.")
    
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: DataFrame still contains {missing_count} missing values.")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for required columns and minimum row count.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has only {len(df)} rows, minimum required is {min_rows}")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None],
        'B': [4, None, 4, 6, 7],
        'C': ['x', 'y', 'x', 'z', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (drop duplicates, fill with mean):")
    cleaned = clean_dataframe(df, drop_duplicates=True, fill_missing='mean')
    print(cleaned)
    
    print("\nValidation result:", validate_dataframe(cleaned, required_columns=['A', 'B']))
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Removed rows with missing values.")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}.")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
        print("Filled missing categorical values with mode.")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has less than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', None, 'z']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nData valid: {is_valid}")
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - df_clean.shape[0]
    
    # Handle missing values
    if columns_to_check is None:
        columns_to_check = df_clean.columns
    
    missing_counts = {}
    for column in columns_to_check:
        if column in df_clean.columns:
            missing_count = df_clean[column].isnull().sum()
            missing_counts[column] = missing_count
            
            if missing_count > 0:
                if fill_strategy == 'mean' and pd.api.types.is_numeric_dtype(df_clean[column]):
                    fill_value = df_clean[column].mean()
                    df_clean[column].fillna(fill_value, inplace=True)
                elif fill_strategy == 'median' and pd.api.types.is_numeric_dtype(df_clean[column]):
                    fill_value = df_clean[column].median()
                    df_clean[column].fillna(fill_value, inplace=True)
                elif fill_strategy == 'mode':
                    fill_value = df_clean[column].mode()[0] if not df_clean[column].mode().empty else None
                    df_clean[column].fillna(fill_value, inplace=True)
                elif fill_strategy == 'drop':
                    df_clean = df_clean.dropna(subset=[column])
                else:
                    # Default: fill with a placeholder
                    df_clean[column].fillna('MISSING', inplace=True)
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    # Print cleaning summary
    print(f"Data cleaning completed:")
    print(f"  - Removed {removed_duplicates} duplicate rows")
    print(f"  - Missing values handled using '{fill_strategy}' strategy")
    
    if missing_counts:
        print("  - Missing values per column:")
        for column, count in missing_counts.items():
            if count > 0:
                print(f"    {column}: {count} missing values")
    
    print(f"  - Final dataset shape: {df_clean.shape}")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate that the DataFrame meets basic requirements.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def main():
    # Example usage
    data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
        'age': [25, 30, None, 40, 50, 50, 35],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0, 95.5]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, fill_strategy='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    try:
        validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
        print("\nData validation passed!")
    except ValueError as e:
        print(f"\nData validation failed: {e}")

if __name__ == "__main__":
    main()