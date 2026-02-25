
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
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, overwrites input file
        subset (list, optional): Columns to consider for identifying duplicates
        keep (str): Which duplicate to keep - 'first', 'last', or False to drop all
    
    Returns:
        int: Number of duplicates removed
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        final_rows = len(df_clean)
        
        duplicates_removed = initial_rows - final_rows
        
        if output_file is None:
            output_file = input_file
        
        df_clean.to_csv(output_file, index=False)
        
        print(f"Removed {duplicates_removed} duplicate rows")
        print(f"Original rows: {initial_rows}, Cleaned rows: {final_rows}")
        print(f"Saved to: {output_file}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return -1
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty")
        return -1
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return -1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)