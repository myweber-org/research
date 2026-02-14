
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
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

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
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

def clean_dataset(df, columns_to_clean=None):
    """
    Clean dataset by removing outliers from specified columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_clean = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original dataset shape:", df.shape)
    print("Original summary for column A:")
    print(calculate_summary_stats(df, 'A'))
    
    cleaned_df = clean_dataset(df, ['A', 'B'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Cleaned summary for column A:")
    print(calculate_summary_stats(cleaned_df, 'A'))import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_method='mean', output_file=None):
    """
    Load a CSV file, handle missing values, and optionally save cleaned data.
    
    Args:
        filepath (str): Path to the input CSV file.
        fill_method (str): Method to fill missing values ('mean', 'median', 'mode', or 'zero').
        output_file (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
    
    Returns:
        pandas.DataFrame or None: Cleaned DataFrame if output_file is None, else None.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            print("Missing values found:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count} missing")
            
            # Fill missing values based on specified method
            for col in df.columns:
                if df[col].isnull().any():
                    if fill_method == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                        fill_value = df[col].mean()
                    elif fill_method == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                        fill_value = df[col].median()
                    elif fill_method == 'mode':
                        fill_value = df[col].mode()[0] if not df[col].mode().empty else np.nan
                    elif fill_method == 'zero':
                        fill_value = 0
                    else:
                        fill_value = df[col].ffill().bfill()  # Fallback to forward/backward fill
                    
                    df[col].fillna(fill_value, inplace=True)
            
            print("Missing values have been filled.")
        else:
            print("No missing values found.")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows.")
        
        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
        
        # Save or return results
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Cleaned data saved to: {output_file}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_numeric_columns(df, columns=None):
    """
    Validate that specified columns contain only numeric values.
    
    Args:
        df (pandas.DataFrame): DataFrame to validate.
        columns (list, optional): List of column names to validate. 
                                 If None, validates all numeric columns.
    
    Returns:
        dict: Validation results with column names as keys and boolean results as values.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    validation_results = {}
    for col in columns:
        if col in df.columns:
            # Check for non-numeric values in numeric columns
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
            validation_results[col] = non_numeric == 0
            if non_numeric > 0:
                print(f"Warning: Column '{col}' contains {non_numeric} non-numeric values.")
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
            validation_results[col] = False
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'value': [10.5, 20.3, None, 15.7, 20.3, None, 12.9, 12.9],
        'category': ['A', 'B', 'A', None, 'B', 'C', 'A', 'A'],
        'score': [85, 92, 78, None, 92, 67, 85, 85]
    }
    
    # Create a test DataFrame
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data('test_data.csv', fill_method='mean', output_file='cleaned_data.csv')
    
    # Validate numeric columns
    if cleaned_df is not None:
        validation = validate_numeric_columns(cleaned_df, ['value', 'score'])
        print(f"Validation results: {validation}")