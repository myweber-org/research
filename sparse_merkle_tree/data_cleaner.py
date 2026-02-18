
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column in a dataset using the IQR method.
    
    Parameters:
    data (numpy.ndarray): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    numpy.ndarray: Dataset with outliers removed from the specified column.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1] or column < 0:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    
    return data[mask]
def remove_duplicates(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for detecting outliers ('iqr', 'zscore')
    columns (list): Specific columns to clean, if None clean all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Determine columns to process
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    # Handle missing values
    for col in columns:
        if col in df_clean.columns and df_clean[col].dtype in [np.float64, np.int64]:
            if missing_strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif missing_strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif missing_strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif missing_strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
    
    # Handle outliers
    for col in columns:
        if col in df_clean.columns and df_clean[col].dtype in [np.float64, np.int64]:
            if outlier_method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
                df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            
            elif outlier_method == 'zscore':
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                z_scores = (df_clean[col] - mean_val) / std_val
                
                # Cap values beyond 3 standard deviations
                df_clean[col] = np.where(z_scores < -3, mean_val - 3*std_val, df_clean[col])
                df_clean[col] = np.where(z_scores > 3, mean_val + 3*std_val, df_clean[col])
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def calculate_statistics(df, columns=None):
    """
    Calculate basic statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Specific columns to analyze
    
    Returns:
    dict: Dictionary containing statistics for each column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = {}
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing_count': df[col].isna().sum(),
                'missing_percentage': (df[col].isna().sum() / len(df)) * 100
            }
    
    return stats

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 5000]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nStatistics:")
    print(calculate_statistics(df))
    
    # Clean the data
    df_clean = clean_dataset(df, missing_strategy='mean', outlier_method='iqr')
    print("\nCleaned DataFrame:")
    print(df_clean)
    print("\nCleaned Statistics:")
    print(calculate_statistics(df_clean))
    
    # Validate
    is_valid, message = validate_dataframe(df_clean, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {is_valid} - {message}")