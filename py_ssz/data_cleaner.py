
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 14, 13, 12, 11]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Remove outliers
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("Cleaned DataFrame (outliers removed):")
    print(cleaned_df)
    print()
    
    # Calculate statistics
    stats = calculate_basic_stats(cleaned_df, 'values')
    print("Statistics for cleaned data:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    # Validate DataFrame
    is_valid, message = validate_dataframe(cleaned_df, ['values'])
    print(f"\nValidation: {message}")import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for duplicates.
            If None, checks all columns. Defaults to None.
        fill_missing (str or value): Method to fill missing values.
            Can be 'mean', 'median', 'mode', or a specific value. Defaults to 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if cleaned_df.isnull().sum().sum() > 0:
        if fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
        else:
            cleaned_df = cleaned_df.fillna(fill_missing)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def get_cleaning_report(df, cleaned_df):
    """
    Generate a report comparing original and cleaned DataFrames.
    
    Args:
        df (pd.DataFrame): Original DataFrame.
        cleaned_df (pd.DataFrame): Cleaned DataFrame.
    
    Returns:
        dict: Dictionary containing cleaning statistics.
    """
    report = {
        'original_rows': len(df),
        'cleaned_rows': len(cleaned_df),
        'duplicates_removed': len(df) - len(cleaned_df),
        'original_missing': df.isnull().sum().sum(),
        'cleaned_missing': cleaned_df.isnull().sum().sum(),
        'columns_processed': list(df.columns)
    }
    
    return report

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'A': [1, 2, 2, 4, None, 6],
#         'B': [10, 20, 20, None, 50, 60],
#         'C': ['x', 'y', 'y', 'z', None, 'x']
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\nMissing values:", df.isnull().sum().sum())
#     
#     # Clean the data
#     cleaned = clean_dataset(df, fill_missing='median')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     # Generate report
#     report = get_cleaning_report(df, cleaned)
#     print("\nCleaning Report:")
#     for key, value in report.items():
#         print(f"{key}: {value}")