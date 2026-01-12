
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {original_shape[0] - cleaned_df.shape[0]} duplicate rows")
    
    # Handle missing values
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values")
        elif fill_missing in ['mean', 'median']:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if fill_missing == 'mean':
                    cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            print(f"Filled missing numeric values with {fill_missing}")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col].fillna(mode_val[0], inplace=True)
            print("Filled missing categorical values with mode")
    
    # Report cleaning results
    final_shape = cleaned_df.shape
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {final_shape}")
    print(f"Rows removed: {original_shape[0] - final_shape[0]}")
    
    return cleaned_df

def validate_dataset(df, required_columns=None, unique_constraints=None):
    """
    Validate a DataFrame for data quality issues.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of columns that must be present
        unique_constraints (list): List of columns that should have unique values
    
    Returns:
        dict: Dictionary containing validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': []
    }
    
    # Check for required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_cols}")
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        validation_results['issues'].append(f"Found {duplicate_count} duplicate rows")
    
    # Check unique constraints
    if unique_constraints:
        for col in unique_constraints:
            if col in df.columns:
                unique_count = df[col].nunique()
                total_count = len(df[col])
                if unique_count < total_count:
                    validation_results['issues'].append(
                        f"Column '{col}' has {total_count - unique_count} duplicate values"
                    )
    
    # Check for missing values
    missing_by_col = df.isnull().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0]
    if not cols_with_missing.empty:
        validation_results['issues'].append(
            f"Missing values in columns: {dict(cols_with_missing)}"
        )
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
        'age': [25, 30, None, 35, 28, 28, 40],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned dataset
    print("\n" + "="*50 + "\n")
    validation = validate_dataset(
        cleaned_df,
        required_columns=['id', 'name', 'age'],
        unique_constraints=['id']
    )
    
    print("Validation Results:")
    print(f"Is Valid: {validation['is_valid']}")
    if validation['issues']:
        print("Issues Found:")
        for issue in validation['issues']:
            print(f"  - {issue}")