
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or array-like): The dataset containing the column to clean.
    column (int or str): The index or name of the column to process.
    
    Returns:
    tuple: A tuple containing:
        - cleaned_data (list): Data with outliers removed.
        - outlier_indices (list): Indices of removed outliers.
    """
    if not data:
        return [], []
    
    # Convert data to numpy array for easier calculations
    data_array = np.array(data)
    
    # Extract the column values
    if isinstance(column, int):
        column_values = data_array[:, column].astype(float)
    else:
        # If column is a string, we'd need a more complex implementation
        # For simplicity, assuming integer index for this example
        column_values = data_array[:, int(column)].astype(float)
    
    # Calculate Q1, Q3 and IQR
    Q1 = np.percentile(column_values, 25)
    Q3 = np.percentile(column_values, 75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify non-outliers
    non_outlier_mask = (column_values >= lower_bound) & (column_values <= upper_bound)
    outlier_indices = np.where(~non_outlier_mask)[0].tolist()
    
    # Filter data to remove outliers
    cleaned_data = [data[i] for i in range(len(data)) if non_outlier_mask[i]]
    
    return cleaned_data, outlier_indices

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (list or array-like): The dataset.
    column (int): The index of the column to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, std, min, and max.
    """
    if not data:
        return {}
    
    data_array = np.array(data)
    column_values = data_array[:, column].astype(float)
    
    stats = {
        'mean': np.mean(column_values),
        'median': np.median(column_values),
        'std': np.std(column_values),
        'min': np.min(column_values),
        'max': np.max(column_values),
        'count': len(column_values)
    }
    
    return stats

# Example usage
if __name__ == "__main__":
    # Sample data
    sample_data = [
        [1, 10.5],
        [2, 12.3],
        [3, 11.8],
        [4, 100.0],  # This is an outlier
        [5, 10.9],
        [6, 11.2],
        [7, 9.8],
        [8, 12.1],
        [9, 200.0],  # This is an outlier
        [10, 11.5]
    ]
    
    print("Original data:")
    for row in sample_data:
        print(row)
    
    # Remove outliers from column 1
    cleaned_data, outliers = remove_outliers_iqr(sample_data, 1)
    
    print(f"\nRemoved {len(outliers)} outliers at indices: {outliers}")
    print("\nCleaned data:")
    for row in cleaned_data:
        print(row)
    
    # Calculate statistics
    stats = calculate_basic_stats(cleaned_data, 1)
    print(f"\nStatistics for cleaned data column 1:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
import pandas as pd
import numpy as np
import sys

def clean_data(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Fill missing numeric values with column mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove outliers using IQR method for numeric columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaning completed. Cleaned data saved to {output_file}")
        print(f"Original rows: {len(pd.read_csv(input_file))}, Cleaned rows: {len(df)}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file.csv> <output_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_data(input_file, output_file)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Strategy to fill missing values. 
            Can be 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, rows with missing values are dropped.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    else:
        cleaned_df = cleaned_df.dropna()
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid} - {message}")