import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df: Input DataFrame
        column_mapping: Dictionary for renaming columns
        drop_duplicates: Whether to drop duplicate rows
        fill_missing: Strategy for filling missing values ('mean', 'median', 'mode', or value)
    
    Returns:
        Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Rename columns if mapping provided
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Drop duplicate rows
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        if removed > 0:
            print(f"Removed {removed} duplicate rows")
    
    # Handle missing values
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().any():
            missing_count = cleaned_df[column].isnull().sum()
            print(f"Column '{column}' has {missing_count} missing values")
            
            if cleaned_df[column].dtype in ['int64', 'float64']:
                if fill_missing == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                else:
                    fill_value = fill_missing
                
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                print(f"Filled missing values in '{column}' with {fill_value}")
            else:
                # For non-numeric columns, use mode or specified value
                if fill_missing in ['mean', 'median', 'mode']:
                    fill_value = cleaned_df[column].mode()[0]
                else:
                    fill_value = fill_missing
                
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                print(f"Filled missing values in '{column}' with '{fill_value}'")
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate the cleaned dataset.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        numeric_columns: List of columns that should be numeric
    
    Returns:
        Boolean indicating if validation passed
    """
    validation_passed = True
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            validation_passed = False
    
    # Check numeric columns
    if numeric_columns:
        for column in numeric_columns:
            if column in df.columns:
                if not np.issubdtype(df[column].dtype, np.number):
                    print(f"Column '{column}' should be numeric but has dtype {df[column].dtype}")
                    validation_passed = False
    
    # Check for remaining missing values
    if df.isnull().any().any():
        missing_cols = df.columns[df.isnull().any()].tolist()
        print(f"Dataset still has missing values in columns: {missing_cols}")
        validation_passed = False
    
    if validation_passed:
        print("Data validation passed successfully")
    else:
        print("Data validation failed")
    
    return validation_passed

# Example usage
if __name__ == "__main__":
    # Create sample data with issues
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
        'age': [25, 30, None, 40, 35, 35, 28],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(
        df, 
        column_mapping={'id': 'user_id', 'name': 'user_name'},
        drop_duplicates=True,
        fill_missing='mean'
    )
    
    print("\n" + "="*50 + "\n")
    print("Cleaned dataset:")
    print(cleaned_df)
    
    # Validate the cleaned data
    print("\n" + "="*50 + "\n")
    required_cols = ['user_id', 'user_name', 'age', 'score']
    numeric_cols = ['age', 'score']
    validate_data(cleaned_df, required_columns=required_cols, numeric_columns=numeric_cols)