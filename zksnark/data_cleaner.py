
import pandas as pd

def clean_dataset(df, fill_method='mean'):
    """
    Remove duplicate rows and fill missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        fill_method (str): Method for filling missing values ('mean', 'median', 'mode', or 'zero').
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Fill missing values
    for column in cleaned_df.select_dtypes(include=['number']).columns:
        if cleaned_df[column].isnull().any():
            if fill_method == 'mean':
                cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            elif fill_method == 'median':
                cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            elif fill_method == 'mode':
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            elif fill_method == 'zero':
                cleaned_df[column].fillna(0, inplace=True)
    
    return cleaned_df

def validate_dataset(df):
    """
    Validate dataset by checking for remaining missing values and duplicates.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        dict: Dictionary containing validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 10, 40, 50],
        'C': [100, 200, 300, 400, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_method='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataset(cleaned)
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")