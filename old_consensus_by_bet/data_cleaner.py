
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def calculate_summary_statistics(dataframe, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': dataframe[column].mean(),
        'median': dataframe[column].median(),
        'std': dataframe[column].std(),
        'min': dataframe[column].min(),
        'max': dataframe[column].max(),
        'count': len(dataframe)
    }
    
    return stats

def clean_dataset(dataframe, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names (optional)
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not process column '{column}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100)
    }
    
    # Add some outliers
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[100] = [101, 500]  # Extreme outlier
    sample_df.loc[101] = [102, -200]  # Negative outlier
    
    print("Original dataset shape:", sample_df.shape)
    print("Original statistics:", calculate_summary_statistics(sample_df, 'value'))
    
    cleaned_df = clean_dataset(sample_df, ['value'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'value'))
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

def load_and_clean_data(filepath):
    try:
        data = pd.read_csv(filepath)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        cleaned_data = clean_dataset(data, numeric_cols)
        return cleaned_data
    except Exception as e:
        print(f"Error processing file: {e}")
        return None
import pandas as pd
import hashlib

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: 'first', 'last', or False to drop all duplicates
    
    Returns:
        Cleaned DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def generate_hash(row):
    """
    Generate a hash for a row to identify duplicates.
    
    Args:
        row: pandas Series representing a row
    
    Returns:
        MD5 hash string
    """
    row_string = str(row.values).encode('utf-8')
    return hashlib.md5(row_string).hexdigest()

def clean_dataset(file_path, output_path=None):
    """
    Main function to clean a dataset by removing duplicates.
    
    Args:
        file_path: path to input CSV file
        output_path: path to save cleaned CSV (optional)
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original dataset shape: {df.shape}")
        
        # Add hash column for duplicate detection
        df['row_hash'] = df.apply(generate_hash, axis=1)
        
        # Remove duplicates based on hash
        df_clean = df.drop_duplicates(subset=['row_hash'], keep='first')
        df_clean = df_clean.drop(columns=['row_hash'])
        
        print(f"Cleaned dataset shape: {df_clean.shape}")
        print(f"Removed {len(df) - len(df_clean)} duplicate rows")
        
        if output_path:
            df_clean.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_dataset(input_file, output_file)
    
    if cleaned_df is not None:
        print("Data cleaning completed successfully")
        print(f"Sample of cleaned data:\n{cleaned_df.head()}")