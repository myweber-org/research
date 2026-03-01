
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = df[column].fillna(df[column].median())
    
    # Remove outliers using z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    for column in numeric_cols:
        if df[column].std() != 0:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = load_and_clean_data(input_file)
        save_cleaned_data(cleaned_df, output_file)
        print(f"Original shape: {pd.read_csv(input_file).shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")
    except FileNotFoundError:
        print(f"Error: {input_file} not found")
    except Exception as e:
        print(f"Error during processing: {str(e)}")import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and removing outliers.
    
    Args:
        df: Input pandas DataFrame
        strategy: Method for missing value imputation ('mean', 'median', 'mode')
        outlier_threshold: Z-score threshold for outlier detection
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values
    for column in df_clean.columns:
        if df_clean[column].isnull().any():
            if strategy == 'mean':
                fill_value = df_clean[column].mean()
            elif strategy == 'median':
                fill_value = df_clean[column].median()
            elif strategy == 'mode':
                fill_value = df_clean[column].mode()[0]
            else:
                fill_value = 0
            
            df_clean[column].fillna(fill_value, inplace=True)
    
    # Remove outliers using Z-score method
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
        df_clean = df_clean[z_scores < outlier_threshold]
    
    # Reset index after filtering
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Tuple of (is_valid, error_message)
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

def calculate_statistics(df):
    """
    Calculate basic statistics for numeric columns.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary containing statistics
    """
    stats = {}
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        stats[column] = {
            'mean': df[column].mean(),
            'median': df[column].median(),
            'std': df[column].std(),
            'min': df[column].min(),
            'max': df[column].max(),
            'count': df[column].count()
        }
    
    return stats

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 60],
        'C': ['x', 'y', 'z', 'x', 'y', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, strategy='mean', outlier_threshold=2)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    # Validate
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
    print(f"Validation: {is_valid} - {message}")
    print("\n")
    
    # Calculate statistics
    stats = calculate_statistics(cleaned_df)
    print("Statistics:")
    for col, values in stats.items():
        print(f"{col}: {values}")