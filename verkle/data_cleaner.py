
def deduplicate_list(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistical measures
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

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list, optional): List of numeric columns to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not process column '{column}': {e}")
    
    return cleaned_dfimport pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Strategy for missing value imputation ('mean', 'median', 'mode')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    
    df_clean = df.copy()
    
    # Handle missing values
    for column in df_clean.columns:
        if df_clean[column].dtype in ['float64', 'int64']:
            if strategy == 'mean':
                fill_value = df_clean[column].mean()
            elif strategy == 'median':
                fill_value = df_clean[column].median()
            elif strategy == 'mode':
                fill_value = df_clean[column].mode()[0]
            else:
                fill_value = 0
            
            df_clean[column].fillna(fill_value, inplace=True)
        else:
            df_clean[column].fillna('Unknown', inplace=True)
    
    # Remove outliers using Z-score method
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
        df_clean = df_clean[z_scores < outlier_threshold]
    
    # Reset index after removing outliers
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=10):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    
    if len(df) < min_rows:
        return False, f"Dataset has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    # Check for completely empty columns
    empty_columns = df.columns[df.isnull().all()].tolist()
    if empty_columns:
        return False, f"Completely empty columns: {empty_columns}"
    
    return True, "Dataset validation passed"

def calculate_statistics(df):
    """
    Calculate basic statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Statistics dataframe
    """
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    stats = pd.DataFrame({
        'mean': numeric_df.mean(),
        'median': numeric_df.median(),
        'std': numeric_df.std(),
        'min': numeric_df.min(),
        'max': numeric_df.max(),
        'missing': numeric_df.isnull().sum()
    })
    
    return stats

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],  # Contains outlier and missing value
        'B': [10, 20, 30, 40, 50, 60],
        'C': ['a', 'b', 'c', 'd', 'e', 'f']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, strategy='mean', outlier_threshold=2)
    print("Cleaned dataset:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    # Validate data
    is_valid, message = validate_data(cleaned_df, required_columns=['A', 'B'], min_rows=3)
    print(f"Validation result: {is_valid}")
    print(f"Validation message: {message}")
    print("\n" + "="*50 + "\n")
    
    # Calculate statistics
    stats = calculate_statistics(cleaned_df)
    print("Dataset statistics:")
    print(stats)