
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_na_threshold=0.5):
    """
    Clean a pandas DataFrame by handling missing values,
    removing duplicates, and standardizing column names.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Standardize column names if mapping is provided
    if column_mapping:
        cleaned_df.rename(columns=column_mapping, inplace=True)
    
    # Convert column names to lowercase and replace spaces with underscores
    cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(' ', '_')
    
    # Remove duplicate rows
    initial_rows = len(cleaned_df)
    cleaned_df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(cleaned_df)
    
    # Calculate missing value percentage for each column
    missing_percent = cleaned_df.isnull().sum() / len(cleaned_df)
    
    # Drop columns with too many missing values
    columns_to_drop = missing_percent[missing_percent > drop_na_threshold].index
    cleaned_df.drop(columns=columns_to_drop, inplace=True)
    
    # For remaining columns with missing values, fill with appropriate values
    for col in cleaned_df.columns:
        if cleaned_df[col].isnull().any():
            if cleaned_df[col].dtype in ['int64', 'float64']:
                # Fill numeric columns with median
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            elif cleaned_df[col].dtype == 'object':
                # Fill categorical columns with mode
                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
            elif cleaned_df[col].dtype == 'bool':
                # Fill boolean columns with False
                cleaned_df[col].fillna(False, inplace=True)
    
    # Remove outliers using IQR method for numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing rows
        cleaned_df[col] = np.where(cleaned_df[col] < lower_bound, lower_bound, cleaned_df[col])
        cleaned_df[col] = np.where(cleaned_df[col] > upper_bound, upper_bound, cleaned_df[col])
    
    # Generate cleaning report
    report = {
        'original_rows': len(df),
        'cleaned_rows': len(cleaned_df),
        'duplicates_removed': duplicates_removed,
        'columns_dropped': list(columns_to_drop),
        'columns_remaining': list(cleaned_df.columns),
        'missing_values_filled': cleaned_df.isnull().sum().sum() == 0
    }
    
    return cleaned_df, report

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'Customer ID': [1, 2, 2, 3, 4, 5, None],
        'First Name': ['John', 'Jane', 'Jane', 'Bob', None, 'Alice', 'Charlie'],
        'Last Name': ['Doe', 'Smith', 'Smith', 'Johnson', 'Brown', 'Wilson', 'Davis'],
        'Age': [25, 30, 30, 35, 40, 150, 28],
        'Salary': [50000, 60000, 60000, None, 70000, 80000, 55000],
        'Active': [True, False, False, True, None, True, False]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    column_mapping = {'Customer ID': 'customer_id'}
    cleaned_df, report = clean_dataset(df, column_mapping)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaning Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    # Validate the cleaned data
    try:
        validate_dataframe(cleaned_df, min_rows=5)
        print("\nData validation passed!")
    except ValueError as e:
        print(f"\nData validation failed: {e}")