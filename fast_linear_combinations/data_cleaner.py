import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing is not None:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        if fill_missing == 'mean':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    mean_val = cleaned_df[col].mean()
                    cleaned_df[col] = cleaned_df[col].fillna(mean_val)
                    print(f"Filled missing values in '{col}' with mean: {mean_val:.2f}")
        
        elif fill_missing == 'median':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    median_val = cleaned_df[col].median()
                    cleaned_df[col] = cleaned_df[col].fillna(median_val)
                    print(f"Filled missing values in '{col}' with median: {median_val:.2f}")
        
        elif fill_missing == 'zero':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)
            print("Filled missing numeric values with 0")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
            print(f"Filled missing values in '{col}' with 'Unknown'")
    
    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    print(f"Dataset validation passed. Shape: {df.shape}")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, None],
        'category': ['A', 'B', 'B', None, 'A', 'C'],
        'score': [85, 92, 92, 78, None, 88]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned)
    
    try:
        validate_dataset(cleaned, required_columns=['id', 'value', 'category'])
    except ValueError as e:
        print(f"Validation error: {e}")