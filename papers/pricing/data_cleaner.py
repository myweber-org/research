
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range (IQR) method.
    
    Args:
        data (np.ndarray): Input data array.
        column (int): Column index to check for outliers.
    
    Returns:
        np.ndarray: Data with outliers removed.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the cleaned data.
    
    Args:
        data (np.ndarray): Input data array.
    
    Returns:
        dict: Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': np.mean(data, axis=0),
        'median': np.median(data, axis=0),
        'std': np.std(data, axis=0)
    }
    return stats

def clean_dataset(data, outlier_columns=None):
    """
    Main function to clean dataset by removing outliers from specified columns.
    
    Args:
        data (np.ndarray): Input data array.
        outlier_columns (list): List of column indices to check for outliers.
    
    Returns:
        tuple: Cleaned data and statistics dictionary.
    """
    if outlier_columns is None:
        outlier_columns = list(range(data.shape[1]))
    
    cleaned_data = data.copy()
    for col in outlier_columns:
        cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    stats = calculate_statistics(cleaned_data)
    return cleaned_data, stats

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = np.random.randn(100, 3) * 10 + 50
    
    print("Original data shape:", sample_data.shape)
    cleaned, statistics = clean_dataset(sample_data, [0, 1, 2])
    print("Cleaned data shape:", cleaned.shape)
    
    for key, value in statistics.items():
        print(f"{key}: {value}")
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Clean a CSV file by handling missing values.
    
    Args:
        input_file (str): Path to input CSV file.
        output_file (str): Path to save cleaned CSV file.
        missing_strategy (str): Strategy for handling missing values.
            Options: 'mean', 'median', 'mode', 'drop'.
    """
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        if missing_strategy == 'drop':
            df_cleaned = df.dropna()
        elif missing_strategy in ['mean', 'median', 'mode']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if df[col].isnull().any():
                    if missing_strategy == 'mean':
                        fill_value = df[col].mean()
                    elif missing_strategy == 'median':
                        fill_value = df[col].median()
                    elif missing_strategy == 'mode':
                        fill_value = df[col].mode()[0]
                    
                    df[col] = df[col].fillna(fill_value)
            
            df_cleaned = df
        else:
            raise ValueError(f"Invalid strategy: {missing_strategy}")
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Cleaned data shape: {df_cleaned.shape}")
        print(f"Cleaned data saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pd.DataFrame): Dataframe to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if df is None or df.empty:
        print("Error: Dataframe is empty or None.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    if df.isnull().any().any():
        print("Warning: Dataframe contains missing values.")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, np.nan, 15.2, np.nan, 20.1],
        'category': ['A', 'B', 'A', np.nan, 'C']
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', 'cleaned_data.csv', 'mean')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, ['id', 'value'])
        print(f"Data validation result: {is_valid}")import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing rows with null values and dropping duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): List of columns to check for nulls. 
                                          If None, checks all columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if df.empty:
        return df
    
    original_shape = df.shape
    
    if columns_to_check is None:
        columns_to_check = df.columns
    
    df_cleaned = df.dropna(subset=columns_to_check)
    
    df_cleaned = df_cleaned.drop_duplicates()
    
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Rows removed: {original_shape[0] - df_cleaned.shape[0]}")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
        'age': [25, 30, 35, None, 40, 40],
        'score': [85.5, 90.0, 78.3, 92.1, 88.7, 88.7]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)